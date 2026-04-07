# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.logger import init_logger
from vllm.tool_parsers.pythonic_tool_parser import (
    PythonicToolParser,
    _UnexpectedAstError,
    _handle_single_tool,
    _compute_tool_delta,
    _make_valid_python
)


logger = init_logger(__name__)

class YuanPythonicToolParser(PythonicToolParser):
    """
    A tool call parser that extends PythonicToolParser with prefix and suffix cleaning.
    
    This parser removes common model-generated prefixes and suffixes that may interfere
    with tool call parsing, such as:
    - Prefix: <tool_calls>
    - Suffixes: </tool_calls>, <|end_of_sentence|>, <eod>
    - Additional whitespace
    
    Used when --enable-auto-tool-choice --tool-call-parser cleaned_pythonic are set
    """

    # Prefixes that need to be removed from the model output
    PREFIXES_TO_REMOVE = [
        "<tool_calls>",
    ]

    # Suffixes that need to be removed from the model output
    SUFFIXES_TO_REMOVE = [
        "</tool_calls>",
        "<|end_of_sentence|>",
        "<eod>",
    ]

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

    def _clean_model_output(self, text: str) -> str:
        """
        Remove configured prefixes, suffixes and trim whitespace from the model output.

        Args:
            text: The raw model output text

        Returns:
            Cleaned text with prefixes/suffixes removed and whitespace trimmed
        """
        cleaned = text.strip()

        # Remove prefixes using split
        for prefix in self.PREFIXES_TO_REMOVE:
            parts = cleaned.split(prefix)
            cleaned = parts[-1].strip()

        # Remove suffixes using split
        for suffix in self.SUFFIXES_TO_REMOVE:
            parts = cleaned.split(suffix)
            cleaned = parts[0].strip()

        return cleaned

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response after cleaning.
        """
        # Clean the model output before processing
        cleaned_output = self._clean_model_output(model_output)
        
        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(
                    cleaned_output,
                    timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
                )
                is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s",
                cleaned_output
            )

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

        try:
            # import ast
            module = ast.parse(cleaned_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                # from vllm.tool_parsers.pythonic_tool_parser import _handle_single_tool
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[
                        _handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=None,
                )
            else:
                raise _UnexpectedAstError(
                    "Tool output must be a list of function calls"
                )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from a streaming response after cleaning.
        """
        # Clean the current text before processing
        cleaned_current_text = self._clean_model_output(current_text)
        
        if not cleaned_current_text.startswith("["):
            return DeltaMessage(content=delta_text)

        try:
            valid_and_added_text = _make_valid_python(cleaned_current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            # import ast
            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                raise _UnexpectedAstError(
                    "Tool output must be a list of function calls"
                )
            
            # from vllm.tool_parsers.pythonic_tool_parser import _handle_single_tool
            tool_calls = [
                _handle_single_tool(e)  # type: ignore
                for e in parsed.elts
            ]

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = (
                    index < len(tool_calls) - 1 or ")]" not in added_text
                )
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    # Function call is incomplete. Withhold the closing bracket.
                    withheld_suffix = withheld_suffix + "}"
                # Strings get single quotes in the model-produced string.
                # JSON requires double quotes.
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = _compute_tool_delta(
                    self.streamed_args_for_tool[index], new_call, index, withheld_suffix
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    if (
                        delta.function is not None
                        and delta.function.arguments is not None
                    ):
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # HACK: serving_chat.py inspects the internal state of tool parsers
            # when determining its final streaming delta, automatically
            # adding autocompleted JSON.
            # These two lines avoid that nonsense while ensuring finish_reason
            # is set to tool_calls when at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            elif not added_text and self.current_tool_id > 0:
                # Return an empty DeltaMessage once the tool calls are all done
                # so that finish_reason gets set.
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

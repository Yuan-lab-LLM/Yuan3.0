question_template = """As an AI assistant, your task is to analyze and restructure given solution text of a question by inserting special markers that indicate verification moments.

Follow these instructions carefully:
### ** Validation Module Definition (`<verify>`): **
   - You will be provided with the question, the final solution of the question, and the content of verification for the final solution.
   - The content of verification will check or to say verify the final solution for multiple times.
   - Your task is to insert the `<verify>` and `</verify>` tags in the content of verification to seperate multiple verification moments.
   - `<verify>` is inserted in the beginning of the verification, and `</verify>` is inserted in the end of the verification. So, the content between `<verify>...</verify>` block marks **a validation of the final solution** to confirm if the final solution correctness.
   - Each `<verify>...</verify>` block must contain a clear verification conclusion for the final answer.(e.g., confirmation of correctness, identified error, Result correct, Not correct).
   - the end of the </verify> is the beginning of the <verify>. And whole content of verification is supposed to be into a `<verify>...</verify>` block.

### **Other Critical Rules:**
   - Preserve all original text for the content after the first answer exactly as it appears.
   - Do not modify sentence structure, wording, code, or math.

Question: 
```
{Question_text}
```

Solution: 
```
{Solution_text}
```

**Input Format:**
```
[the verification content]
```

**Output Format:**
```
[the same verification content with appropriate `<verify>` tags inserted, preserving all content exactly]
```

**Input:**
```
{InputText}
```

**Output:**
"""
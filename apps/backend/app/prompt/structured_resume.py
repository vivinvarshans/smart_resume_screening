PROMPT = """
You are a JSON extraction engine. Convert the following resume text into precisely the JSON schema specified below.

CRITICAL REQUIREMENTS:
- You MUST include ALL fields from the schema, even if they are empty arrays [].
- REQUIRED fields that MUST always be present: "Personal Data", "Experiences", "Projects", "Skills", "Research Work", "Achievements", "Education", "Extracted Keywords"
- If a section has no data, use an empty array: []
- Do not skip any fields from the schema.
- Do not compose any extra fields or commentary.
- Do not make up values for any fields.
- Use "Present" if an end date is ongoing.
- Make sure dates are in YYYY-MM-DD format.
- Do not format the response in Markdown or any other format. Just output raw JSON.
- Ensure the JSON is complete and valid before finishing.

Schema:
```json
{0}
```

Resume:
```text
{1}
```

IMPORTANT: Your response MUST be a complete, valid JSON object with ALL fields from the schema above. 
If you don't have data for a field, use an empty array [] or appropriate default value.
Do NOT truncate the response. Include all sections: Personal Data, Experiences, Projects, Skills, Research Work, Achievements, Education, and Extracted Keywords.
"""

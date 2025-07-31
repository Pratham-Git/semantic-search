import ollama

test_query = "show me a mediocre italian place with valet parking"

temperatures_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]
top_p_values_to_test = [0.1, 0.5, 0.7, 0.9, 1.0]

system_prompt = (
    "You are a restaurant search query parser. Extract information and return valid JSON only.\n\n"
    "Rules:\n"
    "1.  **Price Filters (Specific Numbers):**\n"
    "    - If a user gives a single number (e.g., 'for 1500', 'around 1k'), you MUST set BOTH `cost_min` AND `cost_max` to that exact value.\n"
    "    - If a user gives a range (e.g., 'under 2000', 'between 1000-2000'), set `cost_min` and `cost_max` accordingly.\n"
    "2.  **Sorting Preferences (Vague Terms):**\n"
    "    - If a user uses a vague price term, set the `sort_by` field and leave `cost_min`/`cost_max` as null.\n"
    "    - 'cheap', 'affordable', 'budget' -> `sort_by: 'cost_asc'`\n"
    "    - 'mediocre', 'mid-range', 'average price' -> `sort_by: 'cost_mid'`\n"
    "    - 'expensive', 'fancy', 'luxury' -> `sort_by: 'cost_desc'`\n"
    "3.  **Output Structure:**\n"
    "    - You MUST return a JSON object containing ALL of the following keys, using `null` if a value is not present: `cost_min`, `cost_max`, `sort_by`, `cuisines`, `features`, `name`.\n"
    "4.  **Features:**\n"
    "   - Choose from these terms only: ['5-star dining', 'air condition', 'alcohol served', 'authentic japanese cuisine', 'award winners', 'bar', 'barbeque', 'bars & pubs', 'breakfast buffet', 'buffet', 'cafe', 'dance floor', 'dessert', 'disabled friendly', 'dj', 'eatout', 'exotic cocktails', 'formal attire', 'great breakfasts', 'happy hours', 'healthy food', 'home delivery', 'hookah', 'karaoke', 'kebabs', 'kids allowed', 'live kitchen', 'live music', 'live sports screening', 'luxury dining', 'mall parking', 'microbrewery', 'movies', 'new year', 'nightlife', 'outdoor seating', 'parking', 'pet friendly', 'pocket friendly', 'premium imported ingredients', 'pure veg', 'romantic', 'rooftops', 'sake collection', 'seafood', 'shisha', 'smoking area', 'sports bar', 'stags allowed', 'sunday brunches', 'take-away', 'thali', 'vaccinated staff', 'valet parking', 'vegan', 'wheelchair accessible']\n"
)

print(f"Query: '{test_query}'\n")

for temp in temperatures_to_test:
    print(f"--- Testing with Temperature: {temp} ---")
    for p in top_p_values_to_test:
        try:
            response = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test_query}
                ],
                options={
                    "temperature": temp,
                    "top_p": p
                }
            )

            print(f"  Top_p: {p}")
            print(f"  Output: {response['message']['content'].strip()}")
            print("-" * 25)

        except Exception as e:
            print(f"  An error occurred with temp={temp}, top_p={p}: {e}")
            print("-" * 25)
            

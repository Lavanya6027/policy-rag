import pandas as pd
import requests
import time
from datetime import datetime

def get_sheet_df(sheet_id, gid=0):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url)


# Intialize Questions Data
questions_df = get_sheet_df(
    sheet_id="1eWBbk3Tj7IeOWQUgcRDpMhFrxJ-S87lv9bDCdro1qjg", 
    gid=865538587
)
print(questions_df.head())
questions = questions_df['questions'].tolist()

def run_test(queries, target_url, output_file="ollama_results.csv", to_excel=False):
    results = []  # store query-answer pairs

    for query in queries:
        print(f"\n--- Query: {query} ---")

        start_time = time.time()  # ⏱ start timing
        response = requests.post(
            target_url,
            json={"query": query}  # include model if API needs it
        )
        end_time = time.time()  # ⏱ end timing

        response_time = round(end_time - start_time, 3)  # in seconds

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found")
            no_of_tokens = response.json().get("tokens", 0)
            response_time = response.json().get("response_time", response_time)
            print(f"Answer: {answer}")
        else:
            answer = f"Error: {response.status_code} - {response.text}"
            print(answer)

        print(f"⏱ Response Time: {response_time} sec")

        # append result
        results.append({
            "Query": query,
            "Answer": answer,
            "ResponseTime(sec)": response_time,
            "Token Count": no_of_tokens
        })

    # save results with pandas
    df = pd.DataFrame(results)

    if to_excel:
        df.to_excel(output_file, index=False)
        print(f"\n✅ Results saved to Excel: {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to CSV: {output_file}")

output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
run_test(
    queries=questions,
    target_url="http://localhost:5000/chat",
    output_file=output_file,
    to_excel=False
)

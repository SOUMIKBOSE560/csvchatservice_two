INITIAL_CHAT_QUESTIONS = [
    "What insights can I get from this data?",
    "Can you summarize the dataset?"
]

INITIAL_CHART_QUESTIONS = [
    "Can you provide a visualization of the data?",
     "What are the key trends in the data?"
]

def if_initial_chat_question(query: str) -> bool:
    # Convert the query to lowercase and check if it's in the list (case insensitive)
    if query.lower() in [q.lower() for q in INITIAL_CHAT_QUESTIONS]:
        return True
    return False

def if_initial_chart_question(query: str) -> bool:
    # Convert the query to lowercase and check if it's in the list (case insensitive)
    if query.lower() in [q.lower() for q in INITIAL_CHART_QUESTIONS]:
        return True
    return False

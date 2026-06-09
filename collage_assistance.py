from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ===================== TOOLS =====================

@tool
def attendance_calculator(total_classes: int, attended_classes: int) -> str:
    """Calculate attendance percentage and exam eligibility."""
    percentage = (attended_classes / total_classes) * 100
    status = "Eligible for Exam" if percentage >= 75 else "Not Eligible for Exam"
    return f"Attendance Percentage: {percentage:.2f}%\nExam Eligibility Status: {status}"


@tool
def result_calculator(mark1: int, mark2: int, mark3: int, mark4: int, mark5: int) -> str:
    """Calculate average marks, grade, and pass/fail status."""
    average = (mark1 + mark2 + mark3 + mark4 + mark5) / 5

    if average >= 90:
        grade = "A"
    elif average >= 75:
        grade = "B"
    elif average >= 60:
        grade = "C"
    else:
        grade = "D"

    status = "Pass" if average >= 50 else "Fail"

    return f"Average Marks: {average:.2f}\nGrade: {grade}\\nStatus: {status}"


@tool
def fee_balance_calculator(total_course_fee: float, amount_paid: float) -> str:
    """Calculate pending fee amount."""
    pending = total_course_fee - amount_paid
    return f"Pending Fee Amount: ₹{pending:.2f}"


@tool
def library_fine_calculator(delayed_days: int) -> str:
    """Calculate library fine."""
    return f"Fine Amount: ₹{delayed_days * 5}"


@tool
def hostel_fee_calculator(monthly_hostel_fee: float, months_stayed: int) -> str:
    """Calculate total hostel fee."""
    return f"Total Hostel Fee: ₹{monthly_hostel_fee * months_stayed:.2f}"


students = {
    "101": {"name": "Priya", "department": "CSE", "year": 3},
    "102": {"name": "Rahul", "department": "ECE", "year": 2},
    "103": {"name": "Amit", "department": "ME", "year": 4},
}

@tool
def student_information(student_id: str) -> str:
    """Get student information using student ID."""
    return str(students.get(student_id, "Student ID not found"))


tools = [
    attendance_calculator,
    result_calculator,
    fee_balance_calculator,
    library_fine_calculator,
    hostel_fee_calculator,
    student_information,
]

# ===================== LLM =====================

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# ===================== PROMPT =====================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI-powered College Assistant. Automatically choose and call the correct tool(s).",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# ===================== AGENT =====================

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# ===================== TEST CASES =====================

queries = [
    "I attended 72 classes out of 90. Am I eligible for exams?",
    "My marks are 95, 90, 88, 91 and 87. What is my grade?",
    "My course fee is 50000 and I have paid 35000. How much fee is pending?",
    "I returned a library book 8 days late. What is the fine amount?",
    "Hostel fee is 6000 per month and I stayed for 5 months. Calculate my hostel fee.",
    """I attended 80 classes out of 100.
My marks are 90, 85, 88, 92 and 95.
My course fee is 60000 and I paid 45000.

Provide:
1. Attendance Status
2. Grade
3. Pending Fee""",
    "Show details for student ID 101"
]

for q in queries:
    print("\\n" + "=" * 60)
    print("QUERY:", q)
    print("=" * 60)
    result = agent_executor.invoke({"input": q})
    print(result["output"])

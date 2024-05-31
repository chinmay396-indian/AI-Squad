from langchain_community.llms.ollama import Ollama
from crewai import Agent, Task, Process, Crew

llama3 = Ollama(model="llama3")

developer_agent = Agent(
    llm=llama3,
    role="You are an expert Python coder",
    backstory="You are an expert Python coder having a rich experience of coding since 15 years. you specialized in leading with text, numbers and spreadsheets",
    goal="Your goal is to write an exceptional code which is an automation of reminder mail to my clients to renew my software product license",
    allow_delegation = False,
    verbose= True
)

qa_agent = Agent(
    llm=llama3,
    role="You are an exceptional and expert QA professional expecially for Python codes.",
    backstory="You are an expert QA professional, you are a perfectionist in your work and can catch the bugs easily from the code. You can assume yourself as a Python compiler. you can visualize what can go wrong in the production environment",
    goal="To examine the code written by an expert python developer. Look for bugs and test in such a way that no issue should arise in production environmwnt.",
    allow_delegation = True,
    verbose = True
)

coding_task = Task(
    agent= developer_agent,
    description= """This task is to write a pythonic code to draft and send mail to Sales Manager of the repective clients regarding AISquad product's license renewals.I want you to visualize the data as an excel sheet which has the columns as 'Customer Name','H/W ID','Start_Date','End_Date','Days Remaining'and 'Sales Manager'. Customer Name is the name of the client, H/W ID is the hardware ID of the machine, Start_Date is the starting date of the license, End_Date is the ending date of license, DDays Remaining are the number of days remaining for license to expire and Sales Manager column represents the email id of the sales manager of that repective client. now consider the following license data in csv format for which you have to code: Customer Name,Type,H/W ID,Start_Date,End_Date,Days Remaining,H/W ID,Start_Date,End_Date,Days Remaining,H/W ID,CPP_Start_Date,CPP_End_Date,CPP_Days Remaining,Remarks,Sales Manager
ReliableInsurance,Client,D4BXXXXX-XXXXX-1XXX7-XXXXXX-XXXXXXXX,28-Sep-23,30-Sep-24,122,H4BXXXXX-XXXXX-1XXX7-XXXXXX-XXXXXXXX,28-Sep-23,30-Sep-24,122,D4BXXXXX-XXXXX-1XXX7-XXXXXX-XXXXXXXX,30-Sep-23,30-Sep-24,122,,mohit.shah@reliable.com""",
    expected_output = """An expected output of the python coding task would be a mail sent to the Sales Manager. I am providing an example of the mail below:
    Hello mohit.shah@reliable.com, 
This is a gentle reminder! 
We would like to bring to your attention that the AI Squad license for our esteemed customer Reliable Insurance will be expiring/expired in 122 days on 30-09-2024 with Hardware ID: D4BXXXXX-XXXXX-1XXX7-XXXXXX-XXXXXXXX.
As part of our proactive license management process, we kindly request your prompt confirmation regarding the renewal status for this license. It is crucial for us to receive your timely response to ensure uninterrupted service and continued support for Reliable Insurance.
Your cooperation is vital in maintaining a seamless customer experience. If there are any specific details or considerations that require attention during the renewal process, please do not hesitate to inform us.
We appreciate your swift response, as it will enable us to take further action and guarantee a smooth continuation of services for Reliable Insurance.
Looking forward to your confirmation. 
""")

qa_task = Task(
    agent = qa_agent,
    description="""This task is to analyze the code writen by the developer agent. Test this code thouroughly. check for all the minute details.
    If you find any bugs, construct the suggestions to rectify those bugs and delegate it to the developer to fix it with your suggestions.
    """,
    expected_output = "If any bug arises, then explain the bug properly and explain the suggestions to fix it.",
)

sign_off_task = Task(
    agent = qa_agent,
    description = "This task is to provide the signoff when there are absolutely no bugs or issues in the code and it can be ran sucessfully in production environment directly.",
    expected_output = "This task should output a sign off message with the detailed explaination of the bugs/issue fixed along with a very high confidence score.",

)

AI_Squad_crew = Crew(
    agents = [qa_agent,developer_agent],
    tasks=[coding_task, qa_task, sign_off_task],
    process=Process.sequential,
    verbose=2
)


if __name__=="__main__":
    AI_Squad_crew.kickoff()
    print(AI_Squad_crew)

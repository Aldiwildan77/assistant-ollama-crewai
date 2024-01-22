import os

from crewai import Agent, Crew, Task
from langchain.llms.ollama import Ollama
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"


ollama_llm = Ollama(model="openhermes", tags=["food", "indonesia"])
search_tool = DuckDuckGoSearchRun()


# Define your agents with roles and goals
researcher = Agent(
    role="Business Analyst",
    goal="Identify emerging trends in AI and their impact on the industry",
    backstory="""You work as a business analyst, and expertise in trends analyst especially for selling of FMCG food. Your action is gain insight or information from data analysis to inform decision making of businesses.""",
    verbose=False,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_llm,
)
writer = Agent(
    role="Content Strategist",
    goal="Craft compelling content on tech advancements",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles. You transform complex concepts into compelling narratives.""",
    verbose=False,
    allow_delegation=False,
    llm=ollama_llm,
)

# Create tasks for your agents
task1 = Task(
    description="""Identify indonesian food trends in the last 5 years.
      Your answer should be a list of 5 trends with a short description of each.
    """,
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the most business in Indonesia through food (FMCG).
  Your post should be informative yet accessible, catering to a gain audience.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
    agent=writer,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)

from langchain.prompts import PromptTemplate

prompt_template_questions = """
As an expert in crafting exam preparation materials,
your role is to generate targeted practice questions from the provided text to aid students in their studies. 
Your task is to create a set of questions based on the following passage:

------------
{text}
------------

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = ("""
You are an expert in creating practice questions based on study material.
Your goal is to help a student prepare for an exam.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)
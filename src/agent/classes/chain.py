from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field, create_model
from dotenv import load_dotenv


load_dotenv()

class Chain:
    def __init__(self, llm, system, output = None, human = None):
        self.llm = llm
        self.system = system

        if human is not None:
            self.human = human

            self.prompt_scheme =  ChatPromptTemplate.from_messages(
                [
                    ("system", self.system),
                    ("human",  self.human ),
                ]
            )

            self.output_type, self.output_description = output

            self.output_class = create_model(
                'OutputModel',
                score=(self.output_type, Field(description = self.output_description)) 
            )

            self.structured_output = llm.with_structured_output(self.output_class)

            self.chain = self.prompt_scheme | self.structured_output 

        else:
            self.chain = self.system | llm | StrOutputParser()


    def invoke(self, input):
        return self.chain.invoke(input)

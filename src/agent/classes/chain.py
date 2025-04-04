from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, create_model


from dotenv import load_dotenv


load_dotenv()

class Chain:
    def __init__(self, llm, system, human, output):
        self.llm = llm
        self.system = system
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

    def invoke(self, input):
        return self.chain.invoke(input)

from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():

    information = """
Pramukh Swami Maharaj (born Shantilal Patel; ordained Narayanswarupdas Swami; 7 December 1921 – 13 August 2016) was the guru and Pramukh, or president, of the Bochasanwasi Akshar Purushottam Swaminarayan Sanstha (BAPS), a major branch of the Swaminarayan Sampradaya, a Hindu denomination. BAPS regards him as the fifth spiritual successor of Swaminarayan, following Gunatitanand Swami, Bhagatji Maharaj, Shastriji Maharaj, and Yogiji Maharaj. He was believed by his followers to be in constant communion with Swaminarayan, and ontologically, the manifestation of Akshar, the eternal abode of Swaminarayan.[1]

He received initiation as a Hindu Swami in 1940 from Shastriji Maharaj, the founder of BAPS, who later appointed him as President of BAPS in 1950. Yogiji Maharaj declared Pramukh Swami Maharaj to be his spiritual successor and guru of BAPS, a role he commenced in 1971.

As president of BAPS, he had overseen the growth of BAPS from an organization centered in Gujarat, India, to one spread around the world, maintaining many Hindu mandirs and centers outside of India. He built more than 1,100 Hindu temples, including the Swaminarayan Akshardham temples in New Delhi and Gandhinagar, Gujarat.[2] He had also spearheaded the efforts of BAPS Charities, which is the charitable service organization affiliated with BAPS. He was succeeded as the guru and president of the BAPS Swaminarayan Sanstha by Mahant Swami Maharaj
"""

    summary_template = """
Given the following information, please provide a concise summary that captures the key points and main ideas.
The summary should be clear, coherent, and easy to understand.
{information}

1. Short summary.
2. Pramukh Swami Moto.
"""

    summary_prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt | llm
    result = chain.invoke(input={"information": information})
    print(result.content)

    
if __name__ == "__main__":
    main()

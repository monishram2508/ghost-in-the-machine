import os
import time
import pathlib
from dotenv import load_dotenv
from google import genai as gemini
from pathlib import Path

from httpx import ResponseNotRead
from pydantic_core.core_schema import plain_serializer_function_ser_schema

from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_pair(topic, author):
    generic = plain_para(topic)
    stylized = stylized_para(topic, author)
    return generic, stylized

load_dotenv()
root=Path(__file__).parent.parent
path_topics=root/"data"/"topics"
class1=root/"data"/"generic_ai"
class2=root/"data"/"stylized_ai"

client=gemini.Client(api_key=os.getenv("GEMINI_API_KEY"))

def plain_para(topic):
    prompt=f"Write a paragraph about {topic} consisting of 300-400 words, the paragraph should naturally have this topic, Write only the paragraph, no title or introduction needed."
    return response(prompt)


def stylized_para(topic,author):
    prompt=f"Write a paragraph about {topic} consisting of 300-400 words, in the style of writing that {author} uses. Mimic their style including humor, wit, observational tone, and narrative voice. Write only the paragraph, no title or introduction needed."
    return response(prompt)

def response(prompt):
    for i in range(3):
        try:
            response=client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)
    raise Exception("retried 3 times")

if __name__=="__main__":
    class1.mkdir(parents=True,exist_ok=True)
    class2.mkdir(parents=True,exist_ok=True)
    for file in path_topics.glob("*.txt"):
        base=file.stem.replace("_topics","")
        author=" ".join(base.split("_")[:2]).title()
        topics=file.read_text(encoding="utf-8").strip().split("\n")
        
        print(f"processing {base}")
        
        for topic_idx,topic in enumerate(topics,1):
            from concurrent.futures import ThreadPoolExecutor,as_completed

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures={}

                for i in range(25):
                    generic_file=class1/f"{base}_topic{topic_idx:02d}_{topic}_para{i:03d}.txt"
                    stylized_file=class2/f"{base}_topic{topic_idx:02d}_{topic}_para{i:03d}.txt"

                    if generic_file.exists() and stylized_file.exists():
                        print(f"Topic {topic_idx}/10 ({topic}): {i+1}/50 (skipped)",end="\r")
                        continue

                    futures[executor.submit(generate_pair,topic,author)]=(i,generic_file,stylized_file)

                for future in as_completed(futures):
                    i,generic_file,stylized_file=futures[future]

                    try:
                        generic,stylized=future.result()
                        generic_file.write_text(generic,encoding="utf-8")
                        stylized_file.write_text(stylized,encoding="utf-8")
                        print(f"Topic {topic_idx}/10 ({topic}): {i+1}/50",end="\r")
                    except Exception as e:
                        print(f"\nError at topic {topic_idx}, para {i}: {e}")
            print()
        print(f"completed {base}\n")
    print("done!")

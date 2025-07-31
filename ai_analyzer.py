from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from google.api_core.retry import Retry
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalyzer:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.7):
        # Retry with exponential backoff
        retry_strategy = Retry(
            initial=1.0,           # initial wait: 1s
            maximum=10.0,          # max wait between retries
            multiplier=2.0,        # exponential factor
            deadline=20.0          # total retry duration
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            default_retry=retry_strategy
        )

        self._setup_prompts()

        # Optional: In-memory cache (you can replace with Redis or DB)
        self._cache = {}

    def _setup_prompts(self):
        self.summarize_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Please provide a comprehensive summary of the following text. 
            Focus on the main points, key concepts, and important details.

            Text: {text}

            Summary:
            """
        )

        self.topic_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following text and extract the main topics and themes.
            Return the results as a JSON array of topic objects with 'topic' and 'confidence' fields.

            Text: {text}

            Topics (JSON format):
            """
        )

        self.mcq_prompt = PromptTemplate(
            input_variables=["text", "num_questions"],
            template="""
            Based on the following text, generate {num_questions} multiple choice questions.
            Each question should have 4 options (A, B, C, D) with only one correct answer.

            Text: {text}

            Generate the questions in the following JSON format:
            {{
                "questions": [
                    {{
                        "question": "Question text here?",
                        "options": {{
                            "A": "Option A",
                            "B": "Option B",
                            "C": "Option C",
                            "D": "Option D"
                        }},
                        "correct_answer": "A",
                        "explanation": "Brief explanation of why this is correct"
                    }}
                ]
            }}

            Questions:
            """
        )

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Based on the following context, answer the question accurately and concisely.
            If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

            Context: {context}

            Question: {question}

            Answer:
            """
        )

    def summarize_text(self, text: str) -> str:
        try:
            if not text.strip():
                raise ValueError("Content must not be empty")

            if text in self._cache:
                return self._cache[text]["summary"]

            chain = LLMChain(llm=self.llm, prompt=self.summarize_prompt)
            result = chain.run(text=text)
            self._cache[text] = self._cache.get(text, {})
            self._cache[text]["summary"] = result.strip()
            logger.info("Generated summary successfully")
            return result.strip()
        except Exception as e:
            return self._handle_quota_error(e, "summary")

    def extract_topics(self, text: str) -> List[Dict[str, Any]]:
        try:
            if not text.strip():
                raise ValueError("Content must not be empty")

            if text in self._cache and "topics" in self._cache[text]:
                return self._cache[text]["topics"]

            chain = LLMChain(llm=self.llm, prompt=self.topic_prompt)
            result = chain.run(text=text)

            try:
                # --- JSON parsing improvement ---
                result_strip = result.strip()
                if result_strip.startswith("```json") and result_strip.endswith("```"):
                    result_strip = result_strip[7:-3].strip() # Remove ```json\n and \n```
                
                topics = json.loads(result_strip)
                # ------------------------------

                if isinstance(topics, list):
                    self._cache[text] = self._cache.get(text, {})
                    self._cache[text]["topics"] = topics
                    return topics
                else:
                    logger.warning("Unexpected JSON format for topics: %s", result_strip)
                    return []
            except json.JSONDecodeError as jde:
                logger.warning("Could not parse topics JSON: %s. Raw response: %s", jde, result_strip)
                return []
        except Exception as e:
            return [{"topic": "⚠️ Quota exceeded", "confidence": 0.0}] if self._is_quota_error(e) else self._raise_error(e)

    def generate_mcqs(self, text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        try:
            if not text.strip():
                raise ValueError("Content must not be empty")

            cache_key = f"{text}::mcq::{num_questions}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            chain = LLMChain(llm=self.llm, prompt=self.mcq_prompt)
            result = chain.run(text=text, num_questions=num_questions)

            try:
                # --- JSON parsing improvement ---
                result_strip = result.strip()
                if result_strip.startswith("```json") and result_strip.endswith("```"):
                    result_strip = result_strip[7:-3].strip() # Remove ```json\n and \n```

                data = json.loads(result_strip)
                # ------------------------------

                if isinstance(data, dict) and "questions" in data:
                    self._cache[cache_key] = data["questions"]
                    return data["questions"]
                else:
                    logger.warning("Unexpected JSON format for MCQs: %s", result_strip)
                    return []
            except json.JSONDecodeError as jde:
                logger.warning("Could not parse MCQ JSON: %s. Raw response: %s", jde, result_strip)
                return []
        except Exception as e:
            if self._is_quota_error(e):
                return [{
                    "question": "⚠️ Quota exceeded. Please try again later.",
                    "options": {"A": "", "B": "", "C": "", "D": ""},
                    "correct_answer": "",
                    "explanation": ""
                }]
            return self._raise_error(e)

    def answer_question(self, context: str, question: str) -> str:
        try:
            if not context.strip():
                raise ValueError("Content must not be empty")

            chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
            result = chain.run(context=context, question=question)
            return result.strip()
        except Exception as e:
            return "⚠️ Quota exceeded. Please try again in a few minutes." if self._is_quota_error(e) else self._raise_error(e)

    def analyze_document(self, documents: List[Document]) -> Dict[str, Any]:
        try:
            if not documents:
                return {
                    "summary": "No documents provided for analysis.",
                    "topics": [],
                    "mcqs": [],
                    "document_count": 0,
                    "total_length": 0
                }

            full_text = "\n\n".join([doc.page_content for doc in documents if doc.page_content.strip()])
            if not full_text.strip():
                return {
                    "summary": "No text content found in documents.",
                    "topics": [],
                    "mcqs": [],
                    "document_count": len(documents),
                    "total_length": 0
                }

            summary = self.summarize_text(full_text)
            topics = self.extract_topics(full_text)
            mcqs = self.generate_mcqs(full_text, num_questions=3)

            return {
                "summary": summary,
                "topics": topics,
                "mcqs": mcqs,
                "document_count": len(documents),
                "total_length": len(full_text)
            }
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            # If a quota error propagates up, indicate it here as well
            if self._is_quota_error(e):
                return {
                    "summary": "⚠️ Quota exceeded during analysis. Please wait or upgrade your API plan.",
                    "topics": [],
                    "mcqs": [],
                    "document_count": len(documents) if documents else 0,
                    "total_length": 0
                }
            return {
                "summary": f"Error performing analysis: {str(e)}",
                "topics": [],
                "mcqs": [],
                "document_count": len(documents) if documents else 0,
                "total_length": 0
            }

    def _is_quota_error(self, e: Exception) -> bool:
        # Check for both "429" status code and "quota" string in the error message
        return "429" in str(e) or "quota" in str(e).lower()

    def _handle_quota_error(self, e: Exception, action: str) -> str:
        if self._is_quota_error(e):
            logger.error(f"Quota exceeded during {action}")
            return f"⚠️ Quota exceeded. Please wait or upgrade your API plan."
        logger.error(f"Error during {action}: {str(e)}")
        raise e # Re-raise if it's not a quota error

    def _raise_error(self, e: Exception):
        logger.error(str(e))
        raise e
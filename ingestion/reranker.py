from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING,Any
from abc import ABC, abstractmethod

import httpx
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from config import Settings


logger = logging.getLogger(__name__)

class BaseRanker(ABC):
    @abstractmethod
    async def rerank(self,
                     query:str,
                     candiates:list[dict[str,Any]],
                     top_n :int = 5,
                     ) -> list[dict[str,Any]]:
        pass

    
class OpenAIReranker(BaseRanker):

    _score_prompt ="""
                    Rate the relavence of the following document to the query on a scale of 0 to 10.
                    Reply with interger score between 0 to 10 (example '5').
                    please do not reply with any other text.
                    Query: {query}
                    Document: {text}
                    """

    def __init__(self, settings: Settings):
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None  
        self.settings = settings
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    async def _score_one(self,query:str,candiate:dict[str,Any])->float:

        text=candiate.get("text","") or ""
        image_b64 = candiate.get("image_base64")
        modality = candiate.get("modality","text")

        if modality == "image" and image_b64:
            messages : list[dict[str,Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type":"text",
                            "text":(
                                f"Rate the relavence of the following image (and its caption) to the query on a scale of 0 to 10.\n"
                                f"Reply with interger score between 0 to 10 (example '5').\nQuery: {query}\n"
                                + f"caption :{text} if text else ""\n"
                            ),

                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },

                    ],
                }
            ]
        else:
            prompt = self._score_prompt.format(query=query,text=text)
            messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4,
                temperature=0.0,
            )
            raw=response.choices[0].message.content or "0"
            return float(raw)
        except (ValueError, IndexError):
            logger.warning("Could not parse score from OpenAI response: %r", raw)
            return 0.0
        except Exception as exc:
            logger.error("OpenAI scoring failed for chunk: %s", exc)
            return 0.0

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
        ) -> list[dict[str, Any]]:
        """Score all candidates in parallel then return the top-n."""
        scores = await asyncio.gather(
            *[self._score_one(query, c) for c in candidates]
        )
        scored = [
            {**c, "rerank_score": score} for c, score in zip(candidates, scores, strict=False)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]

class JinaReranker(BaseRanker):

    _API_URL = "https://api.jina.ai/v1/rerank"
    _MODEL = "jina-reranker-m0"

    def __init__(self, settings: Settings):
        if settings.jina_api_key is None:
            raise ValueError(
                "JINA_API_KEY must be set when RERANKER_BACKEND=jina. "
                "Sign up at https://jina.ai to get a free API key."
            )
        self._api_key = settings.jina_api_key.get_secret_value()

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        
        docments : list[dict[str,Any]] = []
        for c in candidates:
            text = c.get("text","")
            image_b64 = c.get("image_base64")
            modality = c.get("modality","text")

            if modality == "image" and image_b64:
                docments.append({
                    "text": text,
                    "image":[image_b64]
                    }
                )
            else:
                docments.append({
                    "text": "text",
                })
        
        payload = {
            "model": self._MODEL,
            "query": query,
            "documents": docments,
            "top_k": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self._API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        reranked: list[dict[str, Any]] = []
        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            reranked.append({**candidates[idx], "rerank_score": score})

        return reranked
    

_BACKENDS: dict[str, type[BaseRanker]] = {
    "openai": OpenAIReranker,
    "jina": JinaReranker
}
     
def get_reranker(settings: Settings) -> BaseRanker:

    backend = settings.reranker_backend.lower()
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown reranker backend: {backend}")
    
    logger.info(f"Initializing reranker with backend: {backend}")
    return _BACKENDS[backend](settings)




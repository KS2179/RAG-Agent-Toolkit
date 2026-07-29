"""
agent/orchestrator.py
Agentic RAG orchestrator: manages the LLM tool-calling loop between local RAG and external MCP tools.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.loader import PipelineConfig
from agent.tools import LOCAL_SEARCH_TOOL_SCHEMA, run_local_search
from agent.mcp_client import MCPClientManager
from store.vector_store import RetrievedChunk


@dataclass
class AgenticAnswer:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[str]
    tool_trace: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]] = field(default_factory=list)

    def pretty_print(self) -> None:
        """Print a formatted answer and tool trace to stdout."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(Panel(self.answer, title="Agentic Answer", border_style="cyan"))
            console.print("\n[bold]Tool Trace:[/bold]")
            for step in self.tool_trace:
                console.print(f"  • Tool: [bold green]{step['tool']}[/bold green] | Args: {step['args']}")
            if self.retrieved_chunks:
                console.print("\n[bold]Local Chunks Used:[/bold]")
                for chunk in self.retrieved_chunks:
                    console.print(f"  • [{chunk.citation_id}] {chunk.source} (score={chunk.score:.4f})")
            if self.web_sources:
                console.print("\n[bold]Web Sources Used:[/bold]")
                for i, src in enumerate(self.web_sources, 1):
                    console.print(f"  • [web:{i}] {src.get('url', 'N/A')}")
        except ImportError:
            print(f"Answer:\n{self.answer}\n")
            print("Tool Trace:", self.tool_trace)


class AgenticRAGPipeline:
    def __init__(
        self,
        retriever: Any,
        mcp_manager: MCPClientManager,
        cfg: PipelineConfig,
    ):
        self.retriever = retriever
        self.mcp_manager = mcp_manager
        self.cfg = cfg
        self._llm = self._init_llm()

    def _init_llm(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            return None

    def answer(self, question: str, model: str = "gpt-4o-mini") -> AgenticAnswer:
        """Synchronous entry point that runs the async agentic loop."""
        return asyncio.run(self._async_answer(question, model=model))

    async def _async_answer(self, question: str, model: str = "gpt-4o-mini") -> AgenticAnswer:
        if self._llm is None:
            raise RuntimeError("LLM client not initialised. Check OPENAI_API_KEY.")

        agent_cfg = self.cfg.agent
        strategy = agent_cfg.strategy
        max_iterations = agent_cfg.max_tool_iterations
        score_threshold = agent_cfg.confidence_threshold

        # Ensure MCP connections are active
        await self.mcp_manager.connect_all()

        external_tools = self.mcp_manager.get_tool_schemas()
        local_tool = LOCAL_SEARCH_TOOL_SCHEMA

        # Determine available tools based on strategy
        if strategy == "rag_only":
            available_tools = [local_tool]
        elif strategy == "auto":
            available_tools = [local_tool] + external_tools
        elif strategy == "rag_then_fallback":
            available_tools = [local_tool]
        else:
            available_tools = [local_tool] + external_tools

        system_prompt = self.cfg.prompts.agent_system or (
            "You are a helpful research assistant. Prefer search_local_documents first. "
            "Use external tools if local context is insufficient or missing."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        tool_trace: List[Dict[str, Any]] = []
        retrieved_chunks: List[RetrievedChunk] = []
        web_sources: List[Dict[str, Any]] = []
        citations: List[str] = []

        iteration = 0
        final_answer_text = ""

        while iteration < max_iterations:
            iteration += 1

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
            }
            if available_tools:
                kwargs["tools"] = available_tools
                kwargs["tool_choice"] = "auto"

            response = self._llm.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # Check if model wants to call tools
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                final_answer_text = message.content or ""
                break

            # Convert message to dict format for context append
            msg_dict = {"role": "assistant", "content": message.content}
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
            messages.append(msg_dict)

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"query": tc.function.arguments}

                if tool_name == "search_local_documents":
                    res = run_local_search(self.retriever, args.get("query", question), score_threshold=score_threshold)
                    tool_result_content = json.dumps(res)
                    summary = f"Found {len(res['chunks'])} chunks (top_score={res['top_score']:.4f})"
                    
                    # Store chunks for citation validation
                    if self.retriever and hasattr(self.retriever, "retrieve"):
                        ret_res = self.retriever.retrieve(args.get("query", question))
                        retrieved_chunks.extend(ret_res.chunks)
                        citations.extend(ret_res.citations)

                    # Check for fallback strategy trigger
                    if strategy == "rag_then_fallback" and res.get("below_confidence_threshold"):
                        for ext_t in external_tools:
                            if ext_t not in available_tools:
                                available_tools.append(ext_t)

                else:
                    res = await self.mcp_manager.call_tool(tool_name, args)
                    tool_result_content = json.dumps(res)
                    summary = str(res.get("result", res.get("error", "")))[:200]

                    # Parse URLs for web citations
                    if "result" in res:
                        urls = re.findall(r'https?://[^\s<>"]+', res["result"])
                        for url in urls:
                            if not any(ws.get("url") == url for ws in web_sources):
                                web_sources.append({"url": url})

                tool_trace.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "args": args,
                    "result_summary": summary,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_content,
                })

        if not final_answer_text and messages:
            # Iteration limit reached without final answer
            last_msg = messages[-1]
            final_answer_text = last_msg.get("content") or "Reached maximum tool iterations without complete answer."

        # Enforce agentic citations
        if hasattr(self.retriever, "enforcer"):
            enforcer = self.retriever.enforcer
            check = enforcer.check_agentic(final_answer_text, retrieved_chunks, web_sources)
            if not check.is_valid:
                final_answer_text = check.answer

        await self.mcp_manager.close_all()

        return AgenticAnswer(
            question=question,
            answer=final_answer_text,
            retrieved_chunks=retrieved_chunks,
            citations=list(set(citations)),
            tool_trace=tool_trace,
            web_sources=web_sources,
        )

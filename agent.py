import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from policy_rag import PolicyRetriever

from tools import (
    reset_password,
    lookup_employee,
    grant_file_access,
    query_hr_database,
    escalate_to_human,
)


load_dotenv()


class PolicyAgent:
    def __init__(
        self,
        index_dir: str = "policy/policy_index",
        log_path: str = "logs/decisions.jsonl",
        employee_data_path: str = "data/employees.json",
    ):
        self.model = os.getenv("OLLAMA_MODEL", "phi3")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        self.retriever = PolicyRetriever()
        self.retriever.load_index(index_dir)

        self.employee_data_path = Path(employee_data_path)
        self.employee_db = self.load_employee_data()

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def load_employee_data(self) -> Dict[str, Any]:
        if not self.employee_data_path.exists():
            raise FileNotFoundError(
                f"Employee data file not found: {self.employee_data_path.resolve()}"
            )

        with open(self.employee_data_path, "r", encoding="utf-8") as f:
            employees = json.load(f)

        return {
            emp["employee_id"]: emp
            for emp in employees
        }

    # -------------------------
    # Public entry point
    # -------------------------

    def handle_request(
        self,
        trust_tier: str,
        employee_id: str,
        request_text: str,
    ) -> Dict[str, Any]:
        trust_tier = trust_tier.strip().lower()

        requester_profile = self.employee_db.get(employee_id)

        if requester_profile is None:
            requester_profile = {
                "employee_id": employee_id,
                "trust_tier": trust_tier.title(),
                "account_type": "unknown",
                "department": None,
                "team": None,
                "title": None,
                "manager_id": None,
                "employment_status": "Unknown",
            }

        retrieved = self.retrieve_policy(trust_tier, request_text)
        interpretation = self.interpret_request(
            trust_tier,
            employee_id,
            request_text,
            retrieved,
            requester_profile,
        )

        interpretation = self.enforce_single_tool(interpretation)

        tool_name = interpretation.get("tool_name")
        tool_args = self.sanitize_tool_args(
            tool_name,
            interpretation.get("tool_args", {})
        )

        decision = interpretation.get("decision", "escalate")
        cited_sections = interpretation.get("cited_sections", [])
        rationale = interpretation.get("rationale", "")

        tool_calls = []
        tool_result = None
        final_answer = ""

        # Hard safety gate: Red users cannot use tools except escalation.
        if trust_tier == "red" and tool_name != "escalate_to_human":
            decision = "deny"
            tool_name = None
            final_answer = (
                "I can’t perform that action for an untrusted session. "
                "Team Red users may receive general policy information, but tool execution is not permitted. "
                "You may contact IT directly or ask me to escalate this. "
                f"Policy cited: {self.format_citations(cited_sections or ['0', '5'])}."
            )

        elif decision == "deny":
            final_answer = self.generate_final_answer(
                trust_tier=trust_tier,
                employee_id=employee_id,
                request_text=request_text,
                decision=decision,
                cited_sections=cited_sections,
                rationale=rationale,
                tool_result=None,
            )

        elif decision == "escalate":
            tool_result = escalate_to_human(
                reason=rationale or "Request requires human review.",
                conversation_summary=f"Trust tier: {trust_tier}. Employee: {employee_id}. Request: {request_text}",
            )
            tool_calls.append({"tool": "escalate_to_human", "args": {"reason": rationale}})
            final_answer = (
                "I’m escalating this to a human IT operator because the request requires review. "
                f"Ticket: {tool_result.get('ticket_id')}. "
                f"Policy cited: {self.format_citations(cited_sections or ['5', '23'])}."
            )

        elif decision == "answer":
            final_answer = self.generate_final_answer(
                trust_tier=trust_tier,
                employee_id=employee_id,
                request_text=request_text,
                decision=decision,
                cited_sections=cited_sections,
                rationale=rationale,
                tool_result=None,
            )

        elif decision == "call_tool":
            allowed, denial_reason = self.pre_tool_safety_check(
                trust_tier=trust_tier,
                tool_name=tool_name,
                tool_args=tool_args,
                request_text=request_text,
            )

            if not allowed:
                decision = "deny"
                final_answer = (
                    f"I can’t complete that request. {denial_reason} "
                    f"Policy cited: {self.format_citations(cited_sections)}."
                )
            else:
                tool_result = self.call_tool(tool_name, tool_args)
                tool_calls.append({"tool": tool_name, "args": tool_args})

                filtered_result = self.filter_tool_output(
                    tool_name=tool_name,
                    tool_result=tool_result,
                    request_text=request_text,
                    trust_tier=trust_tier,
                    employee_id=employee_id,
                    cited_sections=cited_sections,
                )

                final_answer = self.generate_final_answer(
                    trust_tier=trust_tier,
                    employee_id=employee_id,
                    request_text=request_text,
                    decision=decision,
                    cited_sections=cited_sections,
                    rationale=rationale,
                    tool_result=filtered_result,
                )

        else:
            decision = "escalate"
            tool_result = escalate_to_human(
                reason="Agent produced unclear decision.",
                conversation_summary=f"Trust tier: {trust_tier}. Employee: {employee_id}. Request: {request_text}",
            )
            tool_calls.append({"tool": "escalate_to_human", "args": {"reason": "unclear decision"}})
            final_answer = (
                f"I’m escalating this because I couldn’t safely determine the correct action. "
                f"Ticket: {tool_result.get('ticket_id')}. Policy cited: {self.format_citations(['23'])}."
            )

        log_record = {
            "timestamp": time.time(),
            "trust_tier": trust_tier,
            "employee_id": employee_id,
            "requester_profile": requester_profile,
            "request_text": request_text,
            "retrieved_sections": [
                {
                    "section_id": r["section_id"],
                    "section_title": r["section_title"],
                    "score": r["score"],
                }
                for r in retrieved
            ],
            "interpretation": interpretation,
            "decision": decision,
            "tool_calls": tool_calls,
            "raw_tool_result": tool_result,
            "final_answer": final_answer,
        }

        self.write_log(log_record)

        return {
            "decision": decision,
            "answer": final_answer,
            "citations": cited_sections,
            "tool_calls": tool_calls,
            "log": log_record,
        }

    # -------------------------
    # Retrieval
    # -------------------------

    def retrieve_policy(self, trust_tier: str, request_text: str) -> List[Dict[str, Any]]:
        query = f"""
        User request: {request_text}

        Find the functional policy section for this request.
        Focus on account management, employee directory, file access, HR data, escalation, and tool-output filtering.
        """
        # print(query)
        return self.retriever.retrieve(query, top_k=6)

    # -------------------------
    # LLM interpretation
    # -------------------------

    def interpret_request(
        self,
        trust_tier: str,
        employee_id: str,
        request_text: str,
        retrieved: List[Dict[str, Any]],
        requester_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        policy_context = "\n\n".join(
            [
                f"Section {r['section_id']} — {r['section_title']}\n{r['text']}"
                for r in retrieved
            ]
        )
        prompt = f"""
You are the policy decision layer for an IT helpdesk agent.

Decide exactly one: answer, call_tool, deny, or escalate.

Escalate and deny as little as possible.

--------------------------------------------------
Decision order:
1. Identify the functional policy section for the request.
2. Determine if the rule is MAY or MUST NOT.
3. Apply trust tier as a gate (not as justification).
4. If policy says MAY and trust tier allows → call_tool or answer.
5. If policy says MUST NOT → deny.
6. If policy explicitly requires escalation → escalate.
7. If genuinely unclear/high-risk → escalate.
--------------------------------------------------

Tools (choose exactly ONE and return correct arguments):
- reset_password(employee_id: str) → Resets the password for the specified employee and returns
- lookup_employee(query: str) → Looks up employee directory information based on a query string and returns allowed fields.
- grant_file_access(employee_id: str, drive_id: str, access_level: str, duration_days: int or null) → Grants file access and returns details.
- query_hr_database(query_type: "policy" or "individual", employee_id: str or null) → Queries HR database for policy info or individual records (only policy info can be returned).
- escalate_to_human(reason: str, conversation_summary: str) → Escalates to a human operator with a reason and summary, returns a ticket ID.


Trust tier gate:
- Blue: verified employee. May use tools when the functional policy allows it.
- Grey: partially verified. Only low-risk actions allowed; otherwise escalate.
- Red: untrusted. Must not use tools except escalate_to_human.

IMPORTANT:
- Trust tier is ONLY a gate. It does not decide the request.
- NEVER deny or escalate a Blue user unless the functional policy explicitly says MUST NOT or requires escalation.
- Do NOT treat "user-specific information" as restricted unless the policy explicitly prohibits 

User:
Trust tier: {trust_tier}
Requester employee ID: {employee_id}
Requester profile:
{json.dumps(requester_profile, indent=2)}
Request: {request_text}

Retrieved policy:
{policy_context}

Return ONLY valid JSON:
{{
  "requested_action": "short description",
  "functional_section": "section number most directly about the request",
  "needed_tool": "reset_password | lookup_employee | grant_file_access | query_hr_database | escalate_to_human | none",
  "trust_tier_analysis": "brief trust-tier gate analysis",
  "allowing_sections": ["section numbers that allow this"],
  "blocking_sections": ["section numbers that prohibit this"],
  "ambiguity_level": "low | medium | high",
  "decision": "answer | call_tool | deny | escalate",
  "tool_name": "reset_password | lookup_employee | grant_file_access | query_hr_database | escalate_to_human | null",
  "tool_args": {{}},
  "cited_sections": ["functional section first, then trust section if relevant"],
  "rationale": "brief explanation"
}}
"""
        raw = self.call_ollama(prompt)
        return self.parse_json(raw)

    def call_ollama(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {
                    "decision": "escalate",
                    "tool_name": "escalate_to_human",
                    "tool_args": {},
                    "cited_sections": ["23"],
                    "rationale": "Could not parse model decision safely.",
                }
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {
                    "decision": "escalate",
                    "tool_name": "escalate_to_human",
                    "tool_args": {},
                    "cited_sections": ["23"],
                    "rationale": "Could not parse model decision safely.",
                }

    # -------------------------
    # Tool safety + execution
    # -------------------------

    def enforce_single_tool(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures only one valid tool is selected.
        If model returns 'lookup_employee | query_hr_database',
        keep only the first one.
        """

        valid_tools = {
            "reset_password",
            "lookup_employee",
            "grant_file_access",
            "query_hr_database",
            "escalate_to_human",
        }

        raw_tool = interpretation.get("tool_name") or interpretation.get("needed_tool")

        if not raw_tool or raw_tool in {"none", "null"}:
            interpretation["tool_name"] = None
            return interpretation

        raw_tool = str(raw_tool).strip()

        # Handle model output like: "lookup_employee | query_hr_database"
        if "|" in raw_tool:
            raw_tool = raw_tool.split("|")[0].strip()

        # Handle model output like: "lookup_employee, query_hr_database"
        if "," in raw_tool:
            raw_tool = raw_tool.split(",")[0].strip()

        if raw_tool not in valid_tools:
            interpretation["tool_name"] = None
        else:
            interpretation["tool_name"] = raw_tool

        return interpretation
    
    def sanitize_tool_args(self, tool_name: Optional[str], tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fixes missing, wrong, or malformed tool arguments.
        Prevents runtime crashes.
        """

        if not tool_name:
            return {}

        # ---- reset_password ----
        if tool_name == "reset_password":
            return {
                "employee_id": str(tool_args.get("employee_id", "")).strip()
            }

        # ---- lookup_employee ----
        if tool_name == "lookup_employee":
            query = tool_args.get("query")

            # fallback: use employee_id if missing
            if not query:
                query = tool_args.get("employee_id", "")

            return {
                "query": str(query)
            }

        # ---- grant_file_access ----
        if tool_name == "grant_file_access":
            return {
                "employee_id": str(tool_args.get("employee_id", "")).strip(),
                "drive_id": str(tool_args.get("drive_id", "")).strip(),
                "access_level": str(tool_args.get("access_level", "read")).strip(),
                "duration_days": (
                    int(tool_args["duration_days"])
                    if tool_args.get("duration_days") is not None
                    else None
                ),
            }

        # ---- query_hr_database ----
        if tool_name == "query_hr_database":
            query_type = tool_args.get("query_type")

            if query_type not in {"policy", "individual"}:
                query_type = "policy"

            return {
                "query_type": query_type,
                "employee_id": tool_args.get("employee_id"),
            }

        # ---- escalate ----
        if tool_name == "escalate_to_human":
            return {
                "reason": str(tool_args.get("reason", "No reason provided")),
                "conversation_summary": str(
                    tool_args.get("conversation_summary", "No summary provided")
                ),
            }

        return {}

    def pre_tool_safety_check(
        self,
        trust_tier: str,
        tool_name: Optional[str],
        tool_args: Dict[str, Any],
        request_text: str,
    ) -> tuple[bool, str]:
        if trust_tier == "red" and tool_name != "escalate_to_human":
            return False, "Team Red sessions cannot execute tools."

        if trust_tier == "grey" and tool_name in {
            "reset_password",
            "grant_file_access",
            "query_hr_database",
        }:
            return False, "This is too high-risk for a partially verified Team Grey session and must be escalated."

        lowered = request_text.lower()

        if tool_name == "reset_password":
            target = str(tool_args.get("employee_id", "")).lower()
            if "admin" in lowered or "executive" in lowered or "service account" in lowered or target.startswith("svc"):
                return False, "Admin, executive, and service account password resets require human IT Security review."

        if tool_name == "grant_file_access":
            drive = str(tool_args.get("drive_id", "")).lower()
            if "personal" in lowered or "personal" in drive:
                return False, "The agent cannot grant access to another employee’s personal drive."
            if "restricted" in lowered or "legal-hold" in lowered or "legal hold" in lowered:
                return False, "Restricted and legal-hold drives require escalation."
            duration = tool_args.get("duration_days")
            if duration is not None and int(duration) > 7:
                return False, "Temporary cross-team access cannot exceed 7 days."

        if tool_name == "query_hr_database":
            if tool_args.get("query_type") == "individual":
                risky_terms = [
                    "salary",
                    "compensation",
                    "performance",
                    "review",
                    "disciplinary",
                    "bonus",
                    "employment status",
                    "still works",
                    "active",
                ]
                if any(term in lowered for term in risky_terms):
                    return False, "Individual HR records are confidential and cannot be disclosed by the agent."

        return True, ""

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "reset_password":
            return reset_password(**tool_args)

        if tool_name == "lookup_employee":
            return lookup_employee(**tool_args)

        if tool_name == "grant_file_access":
            return grant_file_access(**tool_args)

        if tool_name == "query_hr_database":
            return query_hr_database(**tool_args)

        if tool_name == "escalate_to_human":
            return escalate_to_human(**tool_args)

        raise ValueError(f"Unknown tool: {tool_name}")

    # -------------------------
    # Output filtering
    # -------------------------

    def filter_tool_output(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
        request_text: str,
        trust_tier: str,
        employee_id: str,
        cited_sections: List[str],
    ) -> Dict[str, Any]:
        if tool_name == "lookup_employee":
            allowed_fields = {
                "employee_id",
                "name",
                "department",
                "title",
                "manager",
                "office",
                "work_email",
                "work_phone",
            }
            return {k: v for k, v in tool_result.items() if k in allowed_fields}

        if tool_name == "query_hr_database":
            if tool_result.get("query_type") == "policy":
                return {
                    "query_type": tool_result.get("query_type"),
                    "result": tool_result.get("result"),
                }

            return {
                "error": "Individual HR records cannot be disclosed by the agent."
            }

        if tool_name == "reset_password":
            return {
                "status": tool_result.get("status"),
                "account_type": tool_result.get("account_type"),
                "temp_password": tool_result.get("temp_password"),
                "expires_in": tool_result.get("expires_in"),
                "instruction": "Use the self-service portal to set a permanent password.",
            }

        if tool_name == "grant_file_access":
            return {
                "status": tool_result.get("status"),
                "drive_id": tool_result.get("drive_id"),
                "drive_type": tool_result.get("drive_type"),
                "owning_team": tool_result.get("owning_team"),
                "access_granted": tool_result.get("access_granted"),
                "expires": tool_result.get("expires"),
            }

        return tool_result

    # -------------------------
    # Final response generation
    # -------------------------


    def generate_final_answer(
        self,
        trust_tier: str,
        employee_id: str,
        request_text: str,
        decision: str,
        cited_sections: List[str],
        rationale: str,
        tool_result: Optional[Dict[str, Any]],
    ) -> str:
        citation_text = self.format_citations(cited_sections)

        if decision == "deny":
            return (
                f"I can’t complete that request. {rationale} "
                f"Policy cited: {citation_text}."
            )

        if decision == "answer":
            return (
                f"{rationale} "
                f"Policy cited: {citation_text}."
            )

        if decision == "call_tool":
            if not tool_result:
                return f"The request was completed. Policy cited: {citation_text}."

            if "error" in tool_result:
                return f"I can’t disclose that information. {tool_result['error']} Policy cited: {citation_text}."

            if tool_result.get("query_type") == "policy":
                return f"{tool_result.get('result')} Policy cited: {citation_text}."

            if tool_result.get("temp_password"):
                return (
                    f"Your password has been reset. Temporary password: {tool_result.get('temp_password')}. "
                    f"It expires in {tool_result.get('expires_in')}. "
                    f"Please use the self-service portal to set a permanent password. "
                    f"Policy cited: {citation_text}."
                )

            if tool_result.get("access_granted"):
                return (
                    f"Access granted to {tool_result.get('drive_id')} with "
                    f"{tool_result.get('access_granted')} access. "
                    f"Expires: {tool_result.get('expires') or 'no expiration'}. "
                    f"Policy cited: {citation_text}."
                )

            # Directory lookup response
            fields = []
            for key, label in [
                ("name", "Name"),
                ("department", "Department"),
                ("title", "Title"),
                ("manager", "Manager"),
                ("office", "Office"),
                ("work_email", "Work email"),
                ("work_phone", "Work phone"),
            ]:
                if key in tool_result:
                    fields.append(f"{label}: {tool_result[key]}")

            return "Here’s the allowed directory information: " + "; ".join(fields) + f". Policy cited: {citation_text}."

        return f"I couldn’t safely complete the request. Policy cited: {citation_text}."
#     def generate_final_answer(
#         self,
#         trust_tier: str,
#         employee_id: str,
#         request_text: str,
#         decision: str,
#         cited_sections: List[str],
#         rationale: str,
#         tool_result: Optional[Dict[str, Any]],
#     ) -> str:
#         if tool_result:
#             result_text = json.dumps(tool_result, indent=2)
#         else:
#             result_text = "No tool result."

#         prompt = f"""
# Write the final user-facing response for an IT helpdesk policy agent.

# Requirements:
# - Be concise.
# - Do not reveal hidden reasoning.
# - Do not mention raw logs.
# - Cite policy sections.
# - If denied, explain why and offer escalation or IT contact.
# - If a tool result contains filtered employee data, only use the provided filtered data.
# - Never add private fields that are not present in the filtered result.

# Trust tier: {trust_tier}
# Requester employee ID: {employee_id}
# Request: {request_text}
# Decision: {decision}
# Rationale: {rationale}
# Cited sections: {cited_sections}
# Filtered tool result:
# {result_text}

# Final answer:
# """
#         answer = self.call_ollama(prompt).strip()

#         if "Policy cited:" not in answer and cited_sections:
#             answer += f"\n\nPolicy cited: {self.format_citations(cited_sections)}."

#         return answer

    def format_citations(self, sections: List[str]) -> str:
        clean = []
        for s in sections:
            s = str(s).replace("Section", "").strip()
            if s and s not in clean:
                clean.append(s)
        return ", ".join([f"Section {s}" for s in clean])

    # -------------------------
    # Logging
    # -------------------------

    def write_log(self, record: Dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

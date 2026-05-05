import json
from agent import PolicyAgent


def ask_required(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a value.")


def normalize_trust_tier(value: str) -> str:
    value = value.strip().lower()

    if value in {"blue", "b"}:
        return "Blue"
    if value in {"grey", "gray", "g"}:
        return "Grey"
    if value in {"red", "r"}:
        return "Red"

    raise ValueError("Trust tier must be Blue, Grey, or Red.")


def print_result(result: dict) -> None:
    print("\n" + "=" * 80)
    print("AGENT RESPONSE")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "-" * 80)
    print(f"Decision: {result.get('decision')}")
    print(f"Tool calls: {json.dumps(result.get('tool_calls', []), indent=2)}")
    print(f"Citations: {result.get('citations', [])}")
    print("-" * 80)


def main() -> None:
    print("=" * 80)
    print("Gaggia Policy Agent")
    print("=" * 80)
    print("Type Ctrl+C to quit.\n")

    agent = PolicyAgent()

    while True:
        try:
            trust_tier = normalize_trust_tier(
                ask_required("Trust tier (Blue/Grey/Red): ")
            )
            employee_id = ask_required("Requester employee ID: ")
            request_text = ask_required("Request text: ")

            print("\nRunning agent...\n")

            result = agent.handle_request(
                trust_tier=trust_tier,
                employee_id=employee_id,
                request_text=request_text,
            )


            print_result(result)

            again = input("\nRun another request? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                print("Goodbye.")
                break

            print()

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        except Exception as e:
            print("\nERROR")
            print("-" * 80)
            print(str(e))
            print("-" * 80)
            print(
                "Check that Ollama is running, phi3:medium is pulled, and your policy index exists."
            )


if __name__ == "__main__":
    main()
def reset_password(employee_id: str) -> dict:
    accounts = {
        "EMP-2011": "standard",
        "EMP-4010": "admin",
        "svc-deploy": "service",
        "sysadmin-01": "admin",
    }

    account_type = accounts.get(employee_id, "standard")

    return {
        "status": "success",
        "account_type": account_type,
        "temp_password": "Tmp!x92kL",
        "expires_in": "24 hours"
    }


def lookup_employee(query: str) -> dict:
    return {
        "employee_id": "EMP-1042",
        "name": "Sarah Chen",
        "department": "Engineering",
        "title": "Senior Backend Engineer",
        "manager": "David Kim",
        "office": "Building 3, Floor 2",
        "work_email": "s.chen@gaggia.com",
        "work_phone": "x4521",
        "personal_email": "sarah.chen.personal@gmail.com",
        "personal_phone": "555-0147",
        "home_address": "742 Elm Street, Austin, TX",
        "salary": 158000,
        "performance_rating": "Exceeds Expectations",
        "employment_status": "Active"
    }


def grant_file_access(employee_id: str, drive_id: str, access_level: str, duration_days: int | None) -> dict:
    return {
        "status": "success",
        "drive_id": drive_id,
        "drive_type": "team",
        "owning_team": "Marketing",
        "access_granted": access_level,
        "expires": f"{duration_days} days" if duration_days else None
    }


def query_hr_database(query_type: str, employee_id: str | None) -> dict:
    if query_type == "policy":
        return {
            "query_type": "policy",
            "result": "Gaggia employees receive 20 days of PTO per year, accrued monthly. Unused PTO rolls over up to 5 days."
        }

    return {
        "query_type": "individual",
        "employee_id": employee_id,
        "salary": 158000,
        "bonus_target": "15%",
        "last_review": "2024-03-15",
        "performance_rating": "Exceeds Expectations",
        "disciplinary_actions": []
    }


def escalate_to_human(reason: str, conversation_summary: str) -> dict:
    return {
        "status": "escalated",
        "ticket_id": "ESC-20240315-047",
        "estimated_response": "2 hours"
    }
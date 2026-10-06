"""
evals/golden_dataset.py — verified questions and reference answers
grounded in the current NimbusPay Enterprise Policy Handbook.
"""

GOLDEN_DATASET = [
    {
        "question": "What are the general rules for using corporate devices?",
        "reference_answer": (
            "Users may use company-managed devices and approved applications "
            "for normal business activity. Devices must remain protected, "
            "updated and connected to approved corporate services. Users "
            "should lock the screen when leaving a device unattended, install "
            "software only through the approved software catalogue, keep "
            "approved security tools enabled, avoid unknown removable media, "
            "store company files only in approved repositories, and report "
            "suspicious device behaviour to the service desk or security "
            "mailbox."
        ),
    },
    {
        "question": "What should employees do if they receive a suspicious email?",
        "reference_answer": (
            "They should not click links or open attachments. They should use "
            "the Report Phishing button in the mail client or forward the "
            "message to security-reports@nimbuspay.example, then delete it. "
            "If they clicked a malicious link, they should disconnect from "
            "the network and call the service desk immediately."
        ),
    },
    {
        "question": "What are the response times for acknowledging Severity 1, Severity 2 and Severity 3 alerts?",
        "reference_answer": (
            "Severity 1 alerts must be acknowledged within 15 minutes, "
            "Severity 2 alerts within 30 minutes, and Severity 3 alerts "
            "within 4 hours."
        ),
    },
    {
        "question": "When should an Operator escalate a Severity 1 alert?",
        "reference_answer": (
            "If a Severity 1 alert is not resolved within 30 minutes, the "
            "Operator must escalate it to the on-call Administrator by phone "
            "using the on-call rota."
        ),
    },
    {
        "question": "When may Operators modify the operational maintenance schedule?",
        "reference_answer": (
            "Operators may modify the operational maintenance schedule only "
            "within the approved change windows, which are Tuesday and "
            "Thursday from 22:00 to 02:00 UTC. Every schedule change requires "
            "a change ticket and is logged automatically. No schedule changes "
            "are made during the last three days of a month because of the "
            "month-end processing freeze."
        ),
    },
    {
        "question": "What does the enterprise MFA policy require?",
        "reference_answer": (
            "The current policy requires phishing-resistant authentication "
            "for all staff and hardware security keys for Operators and "
            "Administrators. Only Administrators may change the enterprise "
            "MFA policy, and a policy change requires approval from the "
            "Security Governance Board and 14 days' notice to all staff."
        ),
    },
]

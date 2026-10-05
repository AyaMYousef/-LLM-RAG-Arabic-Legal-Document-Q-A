from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="langfuse-connection-test",
    input={"message": "RAG pipeline test"},
) as span:
    span.update(
        output={"status": "Langfuse tracing works"}
    )

langfuse.flush()

print("Langfuse test trace sent successfully.")
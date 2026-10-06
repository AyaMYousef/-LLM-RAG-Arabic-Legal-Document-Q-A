import json

from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def ask_legal_question(self) -> None:
        payload = {
            "question": "ما هو العقد؟",
        }

        with self.client.post(
            "/ask",
            json=payload,
            name="/ask",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )
                return

            try:
                data = response.json()
            except json.JSONDecodeError:
                response.failure("Response was not valid JSON")
                return

            if not data.get("answer"):
                response.failure("Response does not contain an answer")
import requests
import time
import config


class AssemblySTT:

    def __init__(self):
        self.base_url = "https://api.assemblyai.com"
        self.headers = {
            "authorization": config.ASSEMBLYAI_API_KEY
        }

    def upload(self, filePath):
        with open(filePath, "rb") as f:
            response = requests.post(
                self.base_url + "/v2/upload",
                headers=self.headers,
                data=f
            )
        return response.json()["upload_url"]

    def transcribe(self, filePath):

        print("Uploading audio to AssemblyAI...")

        audio_url = self.upload(filePath)

        print("Transcribing with AssemblyAI...")

        data = {
            "audio_url": audio_url,
            "language_detection": True,
            "speech_models": ["universal-3-pro", "universal-2"]
        }

        response = requests.post(
            self.base_url + "/v2/transcript",
            json=data,
            headers=self.headers
        )

        result = response.json()
        
        # SAFE CHECK
        if "id" not in result:
            raise RuntimeError(f"AssemblyAI Error: {result}")

        transcript_id = result["id"]
        polling_endpoint = f"{self.base_url}/v2/transcript/{transcript_id}"

        while True:
            result = requests.get(polling_endpoint, headers=self.headers).json()

            if result['status'] == 'completed':
                return result['text']

            elif result['status'] == 'error':
                raise RuntimeError(result['error'])

            else:
                time.sleep(2)
import ollama
import tempfile
from vlm_ifc import VLM


class OllamaVLM(VLM):
    def __init__(self, model_name="llava"):
        self.model_name = model_name
        self.conversation = []

    def attach_image(self, image_path):
        """
        Add the path of an image to the latest entry of the conversation list.
        """
        self.conversation[-1]["images"] = [image_path]

    def detach_image(self):
        """
        Remove the image path of the latest entry of the conversation list.
        """
        if "images" in self.conversation[-1].keys():
            del self.conversation[-1]["images"]

    def ask(
        self,
        image,
        prompt="",
    ):
        # Create a temporary file in a context manager
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp_file:
            # Save the image to the temporary file
            image.save(tmp_file.name, format="JPEG")

            # Ensure the file is flushed so it can be read
            tmp_file.flush()

            # Add prompt to the conversation history
            self.conversation.append({"role": "user", "content": prompt})

            self.attach_image(tmp_file.name)

            # Send the image to Ollama's model for analysis
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation,
                options={"temperature": 0.3},
            )
            self.detach_image()

            # Add the model's answer to the conversation history
            answer = response.get("message", {}).get("content", "")
            self.conversation.append({"role": "assistant", "content": answer})

        return answer

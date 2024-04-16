"""OpenAI GPT module."""
from typing import Dict, List, Optional, Tuple, Union

from openai.types.beta.threads import ImageFileContentBlock, TextContentBlock
from openai import OpenAI
from openai.types import FileDeleted

from mltemplate import MltemplateBase
from mltemplate.types import Message
from mltemplate.utils import bytes_to_pil, ifnone


class GPT(MltemplateBase):
    """OpenAI GPT class.

    Args:
        name: Name to give GPT agent.
        instructions: Instructions to pass to the agent before starting the conversation.
        tools: List of enabled tools. `Supported tools <https://platform.openai.com/docs/assistants/tools/tools-beta>`_.
        model: GPT model to use.
        filenames: List of filenames that will be uploaded and made available as reference documents to GPT.

    Example::

        from mltemplate.modules import GPT

        gpt = GPT()
        gpt.message('Write a method that takes two integers and computes their greatest common denominator.')
        result = gpt.message('Use the method you just wrote to compute the GCD of 508012190 and 35967750000.')  # 36890

    """

    name = "GPT"
    default_instructions = "You are a helpful personal assistant. Answer user questions. Write code as necessary."
    default_tools = [{"type": "code_interpreter"}, {"type": "retrieval"}]
    default_model = "gpt-4-1106-preview"

    def __init__(
        self,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        filenames: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(filenames, str):
            filenames = [filenames]
        self.client = OpenAI(api_key=self.config["API_KEYS"]["OPENAI"])

        self._assistant_name = ifnone(name, default=self.name)
        self.instructions = ifnone(instructions, default=self.default_instructions)
        self.tools = ifnone(tools, default=self.default_tools)
        self.model = ifnone(model, default=self.default_model)

        self.assistant = None
        self.thread = None
        self._history: Optional[List[Message]] = None
        self.filenames = filenames
        self.files: Dict[str, str] = {}  # maps filenames to file IDs
        self.reset_chat()

        self.logger.info(f"Created GPT object {id(self)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        deleted_status = {}
        for filename in self.files:
            deleted, status = self.delete_file(filename)
            deleted_status[filename] = {}
            deleted_status[filename]["deleted"] = deleted
            deleted_status[filename]["status"] = status
        self.logger.debug(f"GPT object {id(self)} __exit__()")
        self.logger.debug(f"deleted_status: {deleted_status}")

        deleted = [deleted_status[filename]["deleted"] for filename in self.files]
        if not all(deleted):
            raise RuntimeError(
                f"Unable to delete the following files: "
                f"""{[deleted_status[filename]['status'] for filename in self.files 
                    if not deleted_status[filename]['deleted']]}"""
            )
        self.logger.debug(f"deleted: {deleted}")
        return True

    def reset_chat(self):
        """Reset the chat to its starting state."""
        if self.filenames is not None:
            for filename in self.filenames:
                if filename not in self.files:
                    file = self.upload_file(filename)
                    self.files[filename] = file.id
            self.assistant = self.client.beta.assistants.create(
                name=self._assistant_name,
                instructions=self.instructions,
                tools=self.tools,
                model=self.model,
                file_ids=list(self.files.values()),
            )
        else:
            self.assistant = self.client.beta.assistants.create(
                name=self._assistant_name, instructions=self.instructions, tools=self.tools, model=self.model
            )
        self.thread = self.client.beta.threads.create()
        self._history = []

    def add_message(self, text: str):
        """Add a message to the chat history without sending it for a response."""
        self.client.beta.threads.messages.create(thread_id=self.thread.id, role="user", content=text)

    def message(self, text: str, instructions: Optional[str] = None) -> Message:
        """Send a message to the agent and get a response."""
        self.add_message(text)

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id, assistant_id=self.assistant.id, instructions=instructions
        )
        while run.status not in ["completed", "failed", "expired", "cancelled"]:
            if run.status == "requires_action":
                # TODO
                raise NotImplementedError
            run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
        if run.status != "completed":
            raise RuntimeError(f"Message failed with status: {run.status}")

        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)

        conversation = []
        for message in messages:
            conversation.append(self.parse_message(message_id=message.id))
        conversation.reverse()
        self._history = conversation

        return conversation[-1]

    def handle_requires_action(self, run):
        print(f"run: {run}")
        self.logger.exception("NotImplementedError")
        raise NotImplementedError  # TODO

    def parse_message(self, message_id) -> Message:
        # Retrieve the message object
        message = self.client.beta.threads.messages.retrieve(thread_id=self.thread.id, message_id=message_id)

        # Extract the message content
        message_response = Message(sender=message.role, text="")
        for response in message.content:
            if isinstance(response, ImageFileContentBlock):
                image_bytes = self.client.files.content(response.image_file.file_id).read()
                image = bytes_to_pil(image_bytes)
                message_response.images.append(image)
            elif isinstance(response, TextContentBlock):
                text = response.text
                annotations = []
                for index, annotation in enumerate(text.annotations):
                    if file_citation := getattr(annotation, "file_citation", None):
                        text.value = text.value.replace(annotation.text, f" [{index}]")
                        cited_file = self.client.files.retrieve(file_citation.file_id)
                        annotations.append(f"[{index}] {file_citation.quote} from {cited_file.filename}")
                    elif file_path := getattr(annotation, "file_path", None):
                        cited_file = self.client.files.retrieve(file_path.file_id)
                        image_bytes = self.client.files.content(file_path.file_id).read()
                        message_response.images.append(bytes_to_pil(image_bytes))
                        annotations.append(f"[{index}] image {cited_file.filename}")
                message_response.text += text.value
        return message_response

    def history(self) -> List[Message]:
        return self._history

    def upload_file(self, filename: str):
        # Upload a file with an "assistants" purpose
        file = self.client.files.create(
            file=open(filename, "rb"), purpose="assistants"  # pylint: disable=consider-using-with
        )
        return file

    def delete_file(self, filename: str) -> Tuple[bool, Optional[FileDeleted]]:
        if filename in self.files:
            try:  # Delete the reference from the assistant to the pdf file
                self.client.beta.assistants.files.delete(assistant_id=self.assistant.id, file_id=self.files[filename])
            finally:  # Regardless, actually delete the file
                file_deletion_status = self.client.files.delete(file_id=self.files[filename])
            return file_deletion_status.deleted, file_deletion_status
        else:
            return True, None

    def __call__(self, text: str, **kwargs):
        return self.message(text, **kwargs)

    def __str__(self):
        conv_str = ""
        for message in self._history:
            conv_str += f'{message["role"]}: {message["content"]}\n\n'
        return conv_str

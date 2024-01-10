"""Mltemplate discord client class."""
import argparse
import csv
import logging
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
import requests
from discord.ext import commands, tasks

from mltemplate import Config, MltemplateBase, Registry
from mltemplate.backend.gateway import GatewayServer
from mltemplate.utils import bytes_to_pil, default_logger, ifnone


class DiscordClient(MltemplateBase):
    """Mltemplate Discord Client

    The Mltemplate Discord Client provides a listener for the discord bot associated with the API key listed in the
    config.ini file. It is used by the backend to communicate with the Mltemplate discord server, although it could also
    be used to power a personal discord bot by changing the API key.

    Args:
        gateway_host: The URI of the Mltemplate gateway server. If not given, defaults to the URI specified in the
            config file.
        description: The description of the discord bot.
        command_prefixes: The command prefixes for the discord bot. Defaults to ['>', 'mltemplate '].

    code::

        $ python mltemplate/backend/discord/discord_client.py

    """

    training_requests: List[Dict[str, Any]] = []  # keys are {'request_id', 'ctx'}

    def __init__(
        self,
        gateway_host: str = "http://localhost:8081/",
        description="Mltemplate Discord Agent",
        command_prefixes: Optional[List[str]] = None,
    ):
        super().__init__()
        self.gateway_server = GatewayServer.connection(gateway_host)
        self.description = description
        self.command_prefixes = ifnone(command_prefixes, default=[">", "mltemplate "])

        self.intents = discord.Intents.default()
        self.intents.members = True
        self.intents.message_content = True

        try:
            self.server_commands = self.gateway_server.commands()
        except Exception as err:
            self.logger.exception(f"Error raised in fetching server commands:\n{err}")
            self.server_commands = None
        self.supported_commands = [
            "list_commands",
            "train",
            "registry_summary",
            "list_models",
            "list_experiments",
            "list_runs",
            "best_model_for_experiment",
            "load_model",
            "classify_id",
            "classify_image",
            "logs",
        ]

    def discord_bot(self):
        bot = commands.Bot(description=self.description, command_prefix=self.command_prefixes, intents=self.intents)

        def model_summary(models):
            summary = "All models in the registry:\n"
            summary += f'```{"Model":12} {"Version":12} {"Dataset":12} {"Test Accuracy":15} {"Run ID":30}\n'
            for model in models:
                summary += f'{model["name"]:12} {model["version"]:12} {model["dataset"]:12} '
                summary += f'{model["test_acc"]:.4f}{"":<9} '
                summary += f'{model["run_id"]:30}\n'
            summary += "```\n"
            summary += "The best model for each experiment:\n"
            summary += f'```{"Model":12} {"Version":12} {"Dataset":12} {"TestAccuracy":15} {"Run ID":30}\n'
            for experiment in self.gateway_server.list_experiments():
                model = self.gateway_server.best_model_for_experiment(experiment_name=experiment["name"])
                if model is not None:
                    summary += f'{model["name"]:12} {model["version"]:12} {model["dataset"]:12} '
                    summary += f'{model["test_acc"]:.4f}{"":<9} '
                    summary += f'{model["run_id"]:30}\n'
            summary += "```\n"
            return summary

        @bot.event
        async def on_ready():
            self.logger.info(f"Logged in as {bot.user} (ID: {bot.user.id}).")
            check_training_jobs.start()

        @bot.event
        async def on_message(message):
            self.logger.debug(f"Received message from user {message.author}.")
            # Make sure we do not reply to ourselves
            if message.author.id == bot.user.id:
                return

            # If the message starts with a command prefix, do the command and return
            if any((message.content.startswith(prefix) for prefix in self.command_prefixes)):
                await bot.process_commands(message)
                return

            # Else if the message is a DM maintain a standard chat convo
            if not message.guild:  # message is a DM
                self.logger.debug(f"Message from {message.author} sent to free DM chat. Message: {message.content}")
                try:
                    response = self.gateway_server.chat(text=message.content)
                    max_len = 2000
                    num_chunks = max(ceil(len(response.text) / max_len), 1)
                    for idx in range(num_chunks):
                        start = idx * max_len
                        end = (idx + 1) * max_len
                        await message.channel.send(response.text[start:end])
                    if len(response.images) > 0:
                        for image in response.images:
                            filename = os.path.join(self.config["DIR_PATHS"]["TEMP"], "image.png")
                            image.save(filename)
                            with open(filename, "rb") as image_bytes:
                                discord_image = discord.File(image_bytes)
                                await message.channel.send(file=discord_image)
                            await message.channel.send(file=filename)
                    self.logger.debug(f"Returning DM message from user {message.author}.")
                except discord.errors.Forbidden as err:
                    self.logger.exception(f"Error raised in processing message from user {message.author}:\n{err}")

            # Else if the message mentions us in some way, add a reply on how to use us.
            elif "mltemplate" in message.content.lower():
                self.logger.debug(f"Message from {message.author} sent to channel chat. Message: {message.content}")
                try:
                    response = (
                        f"Hello! If you're trying to chat with me, you can chat with me freely by DMing me. You may "
                        f"also run any of my commands in any channel:\n\n"
                        f"command prefixes: {self.command_prefixes}\n"
                        f"commands: {self.supported_commands}\n\n"
                        f"For example, you may get a summary of the models available for use with:\n"
                        f">registry_summary\n\n"
                        f"You may then load a model with:\n"
                        f">load_model MLP 1\n\n"
                        f"And finally classify an image, either by sample ID:\n"
                        f">classify_id 100\n\n"
                        f"Or by uploading an image and using:\n"
                        f">classify_image\n\n"
                    )
                    await message.reply(response, mention_author=True)
                    self.logger.debug(f"Returning channel message from user {message.author}.")
                except discord.errors.Forbidden as err:
                    self.logger.exception(f"Error raised in processing message from user {message.author}:\n{err}")

        @bot.command()
        async def list_commands(ctx):
            self.logger.debug(f"Received list_commands request from user {ctx.author}.")
            message = "Sure. I know the following commands, and can chat freely through DMs:\n"
            message += " ".join([f"\n\t`>{command}`" for command in self.supported_commands])
            message += "\n\nYou may DM me for further help in using my commands."
            self.logger.debug(f"Returning list_commands request for user {ctx.author}:\n{message}")
            await ctx.send(message)

        @bot.command()
        async def registry_summary(ctx):
            self.logger.debug(f"Received registry_summary request from user {ctx.author}.")
            models = self.gateway_server.models()
            if len(models) == 0:
                summary = "The registry is empty."
                self.logger.debug(f"Returning registry_summary request for user {ctx.author}:\n{summary}")
                await ctx.send(summary)
                return

            summary = model_summary(models)
            summary += "And a CSV with the complete registry summary, including model metrics and parameters:\n"
            await ctx.send(summary)

            filename = os.path.join(self.config["DIR_PATHS"]["TEMP"], "registry_summary.csv")
            Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
            with open(filename, "w+", newline="", encoding="utf-8") as file:
                dict_writer = csv.DictWriter(file, models[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(models)
            self.logger.debug(f"Returning registry_summary request for user {ctx.author}:\n{summary}")
            await ctx.send(file=discord.File(filename))

        @bot.command()
        async def list_models(ctx):
            self.logger.debug(f"Received list_models request from user {ctx.author}.")
            models = self.gateway_server.list_models()
            msg = f'```{"Model":<12} {"Version":<12} {"Run ID":<32}\n'
            for model in models:
                msg += f'{model["model"]:<12} '
                msg += f'{int(model["version"]):<12} '
                msg += f'{model["run_id"]:32}\n'
            msg += "```"
            self.logger.debug(f"Returning list_models request for user {ctx.author}:\n{models}")
            await ctx.send(msg)

        @bot.command()
        async def list_experiments(ctx):
            self.logger.debug(f"Received list_experiments request from user {ctx.author}.")
            experiments = self.gateway_server.list_experiments()
            msg = f'```{"Experiment":12} {"Experiment ID":32}\n'
            for experiment in experiments:
                msg += f'{experiment["name"]:<12} '
                msg += f'{experiment["id"]:<32}\n'
            msg += "```"
            self.logger.debug(f"Returning list_experiments request for user {ctx.author}:\n{msg}")
            await ctx.send(msg)

        @bot.command()
        async def list_runs(ctx):
            self.logger.debug(f"Received list_runs request from user {ctx.author}.")
            runs = self.gateway_server.list_runs()
            print(runs[0]["metrics"].keys())
            msg = f'```{"Run ID":<32} {"Train Accuracy":<15} {"Val Accuracy":<15} {"Test Accuracy":<15}\n'
            for run in runs:
                msg += (
                    f'{run["run_id"]:<32} '
                    f'{run["metrics"].get("train_acc_epoch", 0.):<15.4f} '
                    f'{run["metrics"].get("val_acc_epoch", 0.):<15.4f} '
                    f'{run["metrics"].get("test_acc_epoch", 0.):<15.4f}\n'
                )
            msg += "```"
            self.logger.debug(f"Returning list_runs request for user {ctx.author}:\n{msg}")
            await ctx.send(msg)

        @bot.command()
        async def best_model_for_experiment(ctx, experiment_name: str):
            self.logger.debug(f"Received best_model_for_experiment request from user {ctx.author}.")
            model = self.gateway_server.best_model_for_experiment(experiment_name)
            self.logger.debug(f"Returning best_model_for_experiment request for user {ctx.author}:\n{model}")
            await ctx.send(model)

        @bot.command()
        async def load_model(
            ctx, model: Optional[str] = None, version: Optional[str] = None, run_id: Optional[str] = None
        ):
            self.logger.debug(
                f"Received load_model request from user {ctx.author} with arguments: model={model}, "
                f"version={version}, run_id={run_id}."
            )
            if (model is None or version is None) and run_id is None:
                err_message = "Must specify either (1) model and version or (2) run_id."
                self.logger.error(ValueError(err_message))
                await ctx.send(err_message)
                return
            self.gateway_server.load_model(model=model, version=version, run_id=run_id)
            self.logger.debug(f"Returning load_model request for user {ctx.author}.")
            await ctx.send("The model has been successfully loaded and is ready for use.")

        @bot.command()
        async def classify_id(ctx, idx: int):
            self.logger.debug(f"Received classify_id request from user {ctx.author}.")
            classification = self.gateway_server.classify_id(idx=idx)

            classification["image"].save(os.path.join(self.config["DIR_PATHS"]["TEMP"], "image.png"))
            await ctx.send(file=discord.File(os.path.join(self.config["DIR_PATHS"]["TEMP"], "image.png")))

            msg = f'```Label: {classification["label"]}\n' f'Prediction: {classification["prediction"]}\n'
            msg += "Logits: ["
            for logit in classification["logits"][0]:
                msg += f"{logit:.4f} "
            msg += "]```"
            self.logger.debug(f"Returning classify_id request for user {ctx.author}:\n{msg}")
            await ctx.send(msg)

        @bot.command()
        async def classify_image(ctx):
            self.logger.debug(f"Received classify_image request from user {ctx.author}.")
            attachment_url = ctx.message.attachments[0].url
            file_request = requests.get(attachment_url, timeout=60)
            image = bytes_to_pil(file_request.content)
            classification = self.gateway_server.classify_image(image=image)

            msg = f'```Prediction: {classification["prediction"]}\n'
            msg += "Logits: ["
            for logit in classification["logits"][0]:
                msg += f"{logit:.4f} "
            msg += "]```"
            self.logger.debug(f"Returning classify_id request for user {ctx.author}:\n{msg}")
            await ctx.send(msg)

        @bot.command()
        async def chat(ctx, *, text: str):
            self.logger.debug(f"Received chat request from user {ctx.author}.")
            response = self.gateway_server.chat(text=text)
            max_len = 2000
            num_chunks = max(ceil(len(response.text) / max_len), 1)
            for idx in range(num_chunks):
                start = idx * max_len
                end = (idx + 1) * max_len
                await ctx.send(response.text[start:end])
            if len(response.images) > 0:
                for image in response.images:
                    filename = os.path.join(self.config["DIR_PATHS"]["TEMP"], "image.png")
                    image.save(filename)
                    with open(filename, "rb") as image_bytes:
                        discord_image = discord.File(image_bytes)
                        await ctx.send(file=discord_image)
                    await ctx.send(file=filename)
            self.logger.debug(f"Returning chat request for user {ctx.author}.")

        @bot.command()
        async def train(
            ctx, *, command_line_arguments: Optional[str] = "--config-name train.yaml model=mlp dataset=mnist"
        ):
            self.logger.debug(
                f"Received train request from user {ctx.author} with arguments: {command_line_arguments}."
            )
            self.logger.debug(f"ctx.message.id: {ctx.message.id}")
            self.logger.debug(f"type(ctx.message.id): {type(ctx.message.id)}")
            self.gateway_server.train(request_id=str(ctx.message.id), command_line_arguments=command_line_arguments)
            DiscordClient.training_requests.append({"request_id": ctx.message.id, "ctx": ctx})
            msg = "Sure, I've started working on the training request and will let you know when it finishes."
            await ctx.send(msg)
            # msg = 'Training has finished. The registry has been updated with the training results.\n\n'
            # models = self.gateway_server.models()
            # msg += model_summary(models)
            # self.logger.debug(f'Returning train request for user {ctx.author}:\n{msg}')
            # await ctx.send(msg)

        @tasks.loop(seconds=15.0)
        async def check_training_jobs():
            registry = Registry()
            for training_request in DiscordClient.training_requests:
                request_id = training_request["request_id"]
                ctx = training_request["ctx"]
                run_id = registry.run_id_from_request_id(request_id=str(request_id))
                if run_id is not None:
                    self.logger.debug(f"Training request {request_id} has finished with run_id {run_id}.")
                    DiscordClient.training_requests.remove(training_request)

                    message = await ctx.fetch_message(request_id)
                    self.logger.debug(f"Found message {message} from ctx {ctx}.")

                    msg = "Training has finished. The registry has been updated with the training results.\n\n"
                    models = self.gateway_server.models()
                    msg += model_summary(models)
                    self.logger.debug(f"Returning train request for user {ctx.author}:\n{msg}")

                    max_len = 2000
                    num_chunks = max(ceil(len(msg) / max_len), 1)
                    for idx in range(num_chunks):
                        start = idx * max_len
                        end = (idx + 1) * max_len
                        await message.reply(msg[start:end], mention_author=True)

                else:
                    self.logger.debug(f"Training request {request_id} is still running.")

        @bot.command()
        async def logs(ctx):
            self.logger.debug(f"Received logs request from user {ctx.author}.")
            msg = "Sure, here are the server logs:"
            await ctx.send(msg)
            try:
                await ctx.send(file=discord.File(os.path.join(Config()["DIR_PATHS"]["LOGS"], "discord_logs.txt")))
            except Exception as err:
                self.logger.exception(f"Error raised in sending discord logs:\n{err}")
                raise err
            try:
                await ctx.send(
                    file=discord.File(os.path.join(Config()["DIR_PATHS"]["LOGS"], "gateway_server_logs.txt"))
                )
            except Exception as err:
                self.logger.exception(f"Error raised in sending gateway server logs:\n{err}")
                raise err
            try:
                await ctx.send(
                    file=discord.File(os.path.join(Config()["DIR_PATHS"]["LOGS"], "training_server_logs.txt"))
                )
            except Exception as err:
                self.logger.exception(f"Error raised in sending training server logs:\n{err}")
                raise err
            try:
                await ctx.send(file=discord.File(os.path.join(Config()["DIR_PATHS"]["LOGS"], "train_logs.txt")))
            except Exception as err:
                self.logger.exception(f"Error raised in sending train logs:\n{err}")
                raise err
            self.logger.debug(f"Returning logs request for user {ctx.author}.")

        return bot

    def connect_to_discord(self, bot=None):
        self.logger.debug(f"Received request connect_to_discord for {self.name}.")
        if bot is None:
            bot = self.discord_bot()
        bot.run(self.config["API_KEYS"]["DISCORD"])


def run_discord_client():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_host", type=str, nargs="?", default=Config()["HOSTS"]["GATEWAY_SERVER"])
    opt = parser.parse_args()

    discord_client = DiscordClient(gateway_host=opt.gateway_host)
    discord_client.logger = default_logger(
        name=discord_client.name,
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(Config()["DIR_PATHS"]["LOGS"], "discord_logs.txt"),
    )
    discord_client.connect_to_discord()


if __name__ == "__main__":
    run_discord_client()

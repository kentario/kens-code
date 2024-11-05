from dotenv import load_dotenv
import os

import discord
import logging
import datetime

load_dotenv()

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = 1303195148712153088
WORD_RESET_TIME = datetime.timedelta(seconds=30)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)

@client.event
async def on_ready ():
    print(f'We have logged in as {client.user}')

used_words = {}

@client.event
async def on_message (message):
    # Ignore messages from the bot itself.
    if (message.author == client.user):
        return

    # If any used word has been unused for the WORD_RESET_TIME, it will now be considered unused.
    for word in list(used_words.keys()):
        elapsed_time = datetime.datetime.now(datetime.timezone.utc) - used_words[word]["time"]
        time_remaining = WORD_RESET_TIME - elapsed_time
        if (time_remaining.total_seconds() <= 0):
            del used_words[word]
    
    if (message.content.startswith('!help')):
        help_message = ""
        help_message += 'List of commands:\n!help Print this help message\n\n'
        if (len(used_words) > 0):
            help_message += 'List of unusable words:\n\n'

        for word in used_words.keys():
            time_used = used_words[word]["time"]
            elapsed_time = datetime.datetime.now(datetime.timezone.utc) - time_used
            time_remaining = WORD_RESET_TIME - elapsed_time 
            total_seconds = int(time_remaining.total_seconds())
            
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            help_message += f'{word} - Used by {used_words[word]["user"].name} at {time_used}. Resets in {days} days, {hours} hours, {minutes} minutes and {seconds} seconds.\n'

        # If the message gets too long, send it in chunks.
        await message.channel.send(help_message)

    # Make all words lower case so that they are easier to deal with later.
    words = [word.lower() for word in message.content.split()]

    if (len(words) != 3):
        return

    members = message.channel.members

    # A valid name is any part of a username or nick, seperated by whitespace.
    valid_names = []
    for member in members:
        # Convert the names to lowercase using list comprehension.
        valid_names += [name.lower() for name in member.name.split()]
        if (member.nick):
            valid_names += [name.lower() for name in member.nick.split()]

    # The list of all names that are also in the message.
    names_in_message = set(words) & set(valid_names)
    # A message must contain at least one name to be valid.
    if (len(names_in_message) < 1):
        await message.channel.send(f'Invalid message, message {message.content} does not contain any usernames of people in this channel.')
        return

    # Remove all names from the message to make it easier to deal with.
    for name in names_in_message:
        words.remove(name)
    
    # Check if all the words are valid.
    for word in words:
       if (word in used_words.keys()):
           await message.channel.send(f'Invalid message, {word} was already used by {used_words[word]["user"].name} at {used_words[word]["time"]}.')
           return

    # Add each word to the list of used words.
    for word in words:
        used_words[word] = {
            "user": message.author,
            "time": message.created_at
            }

client.run(BOT_TOKEN, log_handler=handler)

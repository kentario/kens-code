from dotenv import load_dotenv
import os

import sqlite3

import discord
import logging
import datetime

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WORD_RESET_TIME = datetime.timedelta(seconds=30)

handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
bot = discord.Bot(intents=intents)

# Set up the database connection.
connection = sqlite3.connect("used_words.db")
cursor = connection.cursor()

# Create a table if it doesn't exist.
cursor.execute("""
    CREATE TABLE IF NOT EXISTS used_words (
        word TEXT PRIMARY KEY,
        user TEXT,
        time TEXT
    )
""")
connection.commit()

def update_used_words ():
    # If any used word has been unused for the WORD_RESET_TIME, it will now be considered unused.
    now = datetime.datetime.now(datetime.timezone.utc)
    cursor.execute("SELECT word, time FROM used_words")
    for word, time in cursor.fetchall():
        elapsed_time = now - datetime.datetime.fromisoformat(time)
        if (elapsed_time >= WORD_RESET_TIME):
            cursor.execute("DELETE FROM used_words WHERE word = ?", (word,))
    connection.commit()
            
@bot.event
async def on_ready ():
    print(f"We have logged in as {bot.user}")
    
@bot.slash_command()
async def help (context):
    update_used_words()
    cursor.execute("SELECT word, user, time FROM used_words")
    words = cursor.fetchall()
    
    help_message = "List of commands:\n/help Print this help message\n\n"

    if (len(words) > 0):
        help_message += "List of unusable words:\n"
        
    for word, user, time in words:
        time_used = datetime.datetime.fromisoformat(time)
        elapsed_time = datetime.datetime.now(datetime.timezone.utc) - time_used
        time_remaining = WORD_RESET_TIME - elapsed_time 
        total_seconds = int(time_remaining.total_seconds())
        
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        help_message += f"{word} - Used by {user} at {time_used}. Resets in {days} days, {hours} hours, {minutes} minutes and {seconds} seconds.\n"
        
    # TODO: If the message gets too long, send it in chunks.
    await context.respond(help_message)
    
    
@bot.event
async def on_message (message):
    # Ignore messages from the bot itself.
    if (message.author == bot.user):
        return

    update_used_words()
    
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
    # If a message doesn't contain any names, it is assumed not to be a congratulation.
    if (len(names_in_message) < 1):
        return

    # Remove all names from the message to make it easier to deal with.
    for name in names_in_message:
        words.remove(name)
    
    # Check if all the words are valid.
    for word in words:
        cursor.execute("SELECT user, time FROM used_words WHERE word = ?", (word,))
        result = cursor.fetchone()
        if (result):
            user, time = result
            await message.add_reaction("❌")
            await message.channel.send(f"Invalid message, {word} was already used by {user} at {time}")
            return

    # Add each word to the list of used words.
    for word in words:
        cursor.execute("INSERT OR REPLACE INTO used_words (word, user, time) VALUES (?, ?, ?)", (word, message.author.name, message.created_at.isoformat()))

    await message.add_reaction("✅")

    connection.commit()

bot.run(BOT_TOKEN)

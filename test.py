import os
from uuid import uuid4
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.agent import RunResponse
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st

# YouTube transcript extraction
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    st.warning("youtube-transcript-api is not installed. Please install it for YouTube support.")
    YouTubeTranscriptApi = None

# Streamlit Page Setup
st.set_page_config(page_title="📰/📺 ➡️ 🎙️ Blog/YouTube to Podcast Agent", page_icon="🎙️")
st.title("📰/📺 ➡️ 🎙️ Blog/YouTube to Podcast Agent")

# Sidebar: API Keys
st.sidebar.header("🔑 API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")

# Check if all keys are provided
keys_provided = all([openai_api_key, elevenlabs_api_key, firecrawl_api_key])

# Input: Blog or YouTube URL
url = st.text_input("Enter the Blog or YouTube URL:", "")

# Button: Generate Podcast
generate_button = st.button("🎙️ Generate Podcast", disabled=not keys_provided)

if not keys_provided:
    st.warning("Please enter all required API keys to enable podcast generation.")

def is_youtube_url(url):
    """Check if the provided URL is a YouTube link."""
    return ("youtube.com/watch" in url) or ("youtu.be/" in url)

def get_youtube_video_id(url):
    """Extract the video ID from a YouTube URL."""
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None

def fetch_youtube_transcript(url):
    """Fetch transcript text from a YouTube video."""
    if not YouTubeTranscriptApi:
        raise ImportError("youtube-transcript-api is not installed.")
    video_id = get_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")
    # Fetch transcript (may fail on cloud platforms without proxy)
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript_list])
    return transcript_text

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a URL first.")
    else:
        # Set API keys as environment variables for Agno and tools
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
        os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

        with st.spinner("Processing... Scraping content, summarizing and generating podcast 🎶"):
            try:
                # Decide content extraction method based on URL type
                if is_youtube_url(url):
                    # Handle YouTube video: extract transcript
                    st.info("Detected YouTube URL. Extracting transcript...")
                    try:
                        content_text = fetch_youtube_transcript(url)
                        if not content_text.strip():
                            st.error("No transcript found for this YouTube video.")
                            st.stop()
                    except Exception as yt_ex:
                        st.error(f"Failed to fetch YouTube transcript: {yt_ex}")
                        st.stop()
                else:
                    # Handle blog/article: use Firecrawl to scrape
                    st.info("Detected blog/article URL. Scraping content...")
                    # We'll let the agent use FirecrawlTools to scrape the content

                # Build agent instructions
                instructions = [
                    "When the user provides a URL:",
                    "1. If it's a blog/article, use FirecrawlTools to scrape the content.",
                    "2. If it's a YouTube video, use the provided transcript text as the content.",
                    "3. Create a concise summary of the content that is NO MORE than 2000 characters long.",
                    "4. The summary should capture the main points while being engaging and conversational.",
                    "5. Use the ElevenLabsTools to convert the summary to audio.",
                    "Ensure the summary is within the 2000 character limit to avoid ElevenLabs API limits.",
                ]

                # Prepare agent tools
                tools = [
                    ElevenLabsTools(
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                        target_directory="audio_generations",
                    ),
                    FirecrawlTools(),
                ]

                # Initialize the Agno Agent
                blog_to_podcast_agent = Agent(
                    name="Blog/YouTube to Podcast Agent",
                    agent_id="blog_to_podcast_agent",
                    model=OpenAIChat(id="gpt-3.5-turbo"),
                    tools=tools,
                    description="You are an AI agent that can generate audio using the ElevenLabs API.",
                    instructions=instructions,
                    markdown=True,
                    debug_mode=True,
                )

                # Run the agent with the appropriate input
                if is_youtube_url(url):
                    # For YouTube, pass the transcript directly
                    agent_input = (
                        f"Convert the following YouTube transcript to a podcast:\n\n{content_text}"
                    )
                else:
                    # For blogs, let the agent handle scraping via FirecrawlTools
                    agent_input = f"Convert the blog content to a podcast: {url}"

                podcast: RunResponse = blog_to_podcast_agent.run(agent_input)

                # Ensure output directory exists
                save_dir = "audio_generations"
                os.makedirs(save_dir, exist_ok=True)

                # Handle audio output
                if podcast.audio and len(podcast.audio) > 0:
                    filename = f"{save_dir}/podcast_{uuid4()}.wav"
                    write_audio_to_file(
                        audio=podcast.audio[0].base64_audio,
                        filename=filename
                    )

                    st.success("Podcast generated successfully! 🎧")
                    audio_bytes = open(filename, "rb").read()
                    st.audio(audio_bytes, format="audio/wav")

                    st.download_button(
                        label="Download Podcast",
                        data=audio_bytes,
                        file_name="generated_podcast.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("No audio was generated. Please try again.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Streamlit app error: {e}")


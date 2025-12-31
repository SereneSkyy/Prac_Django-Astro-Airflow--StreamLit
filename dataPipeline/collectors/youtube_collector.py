from googleapiclient.discovery import build
from googleapiclient.errors import HttpError # For catching specific API errors
from datetime import datetime, timedelta
import os
from .base_collector import BaseCollector
from typing import List


class YouTubeNepal(BaseCollector):
    def __init__(self,api_key:str):
        super().__init__("youtube")
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        
    def _handle_quota_error(self, e):
        if e.resp.status in [403, 429] and "quotaExceeded" in str(e):
            print("CRITICAL: Youtube API Quota Limit Exceeded!")
            print("Check Google Cloud Console. ETL will resume once quota resets")
            return True
        return False

    def search_videos(self, query, max_results: int = 50) -> List[str]:
        """
        Returns up to max_results video IDs that are:
        - published after 2024-01-01
        """
        try:
            print(f"[YT] Searching for: {query}")

            vid_ids = []
            page_token = None

            while True:
                response = self.youtube.search().list(
                    # q=query,
                    q=f"\"{query}\" -shorts",
                    part="id,snippet",
                    type="video",
                    order="relevance",
                    # publishedAfter="2024-01-01T00:00:00Z",
                    maxResults=min(50, max_results - len(vid_ids)),
                    regionCode="NP",
                    pageToken = page_token,
                ).execute()
                vid_ids.extend(
                    item["id"]["videoId"]
                    for item in response.get("items", [])
                )
                page_token = response['nextPageToken']

                if not page_token or len(vid_ids) >= max_results:
                    break

            print(f"[YT] Found {len(vid_ids)} video IDs")
            return vid_ids[:max_results]
        
        except HttpError as e:
            self._handle_quota_error(e) # check for quota here
            print(f"[YT] SEARCH HTTP ERROR: {e}")
            return []
        except Exception as e:
            print(f"[YT] SEARCH ERROR: {e}")
            return []

    def fetch_data(self, video_id, cmt_per_vid: int = 500):
        print(f"[YT] ---> STARTING FETCH FOR VIDEO: {video_id}") # LOG TEST
        try:
            comments = []
            page_token = None
            while True:
                response = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, cmt_per_vid - len(comments)),
                    textFormat="plainText",
                    order='relevance', 
                    pageToken = page_token,
                ).execute()
                comments.extend(response.get("items", []))

                page_token = response.get("nextPageToken")
                if not page_token or len(comments) >= cmt_per_vid:
                    break

            print(f"[YT] Collected {len(comments)} comments for {video_id}")
            return comments

        except HttpError as e:
            if self._handle_quota_error(e): # Check for quota here
                return []
            if e.resp.status == 403:
                print(f"[YT] SKIPPING: Comments are disabled for video {video_id}")
            else:
                print(f"[YT] API ERROR ({e.resp.status}): {e}")
            return []
        except Exception as e:
            print(f"[YT] UNKNOWN ERROR: {e}")
            return []
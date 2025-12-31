from googleapiclient.discovery import build
from googleapiclient.errors import HttpError # For catching specific API errors
from datetime import datetime, timedelta
import os
from .base_collector import BaseCollector


class YouTubeNepal(BaseCollector):
    def __init__(self,api_key:str):
        super().__init__("youtube")
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        
            

    from typing import List

    def _handle_quota_error(self, e):
        if e.resp.status in [403, 429] and "quotaExceeded" in str(e):
            print("CRITICAL: Youtube API Quota Limit Exceeded!")
            print("Check Google Cloud Console. ETL will resume once quota resets")
            return True
        return False
            

    def search_videos(self, query, max_results: int = 1) -> List[str]:
        """
        Returns up to max_results video IDs that are:
        - published after 2024-01-01
        """
        try:
            print(f"api: {self.youtube}")
            print(f"[YT] Searching for: {query}")

            request = self.youtube.search().list(
                # q=query,
                q=f"\"{query}\" -shorts",
                part="id,snippet",
                type="video",
                order="relevance",
                # publishedAfter="2024-01-01T00:00:00Z",
                maxResults=max_results,
                regionCode="NP",
            )
            response = request.execute()

            vid_ids = [item["id"]["videoId"] for item in response.get("items", [])][:max_results]
            print(f"[YT] Found {len(vid_ids)} candidate video IDs: {vid_ids}")
            return vid_ids
        
        except HttpError as e:
            self._handle_quota_error(e) # check for quota here
            print(f"[YT] SEARCH HTTP ERROR: {e}")
            return []
        except Exception as e:
            print(f"[YT] SEARCH ERROR: {e}")
            return []

    def fetch_data(self, video_id, cmt_per_vid: int = 1):
        print(f"[YT] ---> STARTING FETCH FOR VIDEO: {video_id}") # LOG TEST
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=cmt_per_vid,
                textFormat="plainText",
                order='relevance', 
            )
            response = request.execute()
            
            print(f'Comments for {video_id}: {len(response)}')
            return response

        except HttpError as e:
            if self._handle_quota_error(e): # Check for quota here
                return {}
            
            if e.resp.status == 403:
                print(f"[YT] SKIPPING: Comments are disabled for video {video_id}")
            else:
                print(f"[YT] API ERROR ({e.resp.status}): {e}")
            return {}
        except Exception as e:
            print(f"[YT] UNKNOWN ERROR: {e}")
            return {}
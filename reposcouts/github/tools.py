# Design tool for Github repo searching
# Date: 2026-09-06
# Author: zKarina

# Using Github REST API to search for repos info.

import os
import json
import base64

import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

GITHUB_API_BASE_URL = "https://api.github.com"

def _get_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")

    headers = {
        "Accept": "application/vnd.github+json",    # return in JSON
        "X-GitHub-Api-Version": "2026-03-10",       # the version of REST API
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


@tool
def search_github_repositories(
        query: str, 
        limit: int = 5, 
) -> list[dict]: 
    """
    Search GitHub repositories using a GitHub search query.

    Use this tool when you need to discover candidate repositories
    related to a user's technical requirements.

    Args:
        query:
            User Github repository search query.

        limit: 
            Maximum number of repositories to return.
    """

    url = f"{GITHUB_API_BASE_URL}/search/repositories"

    # Search parameters: sort by stars, descending order, limit results
    params = {
        "q": query,
        # "sort": "stars",
        # "order": "desc",
        "per_page": limit,
    }

    # exactly url with request_headers and params.
    response = requests.get(
        url, 
        headers=_get_headers(), 
        params=params, 
        timeout=15, 
    )

    response.raise_for_status()

    data = response.json()

    repositories = []
    for item in data.get("items", []):
        # select only the relevant fields from the repo.
        repositories.append(
            {
                "name": item["name"],
                "full_name": item["full_name"],
                "description": item.get("description"),
                "html_url": item["html_url"],
                "stars": item["stargazers_count"],
                "forks": item["forks_count"],
                "language": item.get("language"),
                "updated_at": item["updated_at"],
            }
        )

    return repositories


@tool
def get_repository_info(
        full_name: str, 
) -> dict:
    """
    Get detailed metadata about one GitHub repository.

    The repository name must use the owner/repository format.

    Args:
        full_name:
            Repository identifier in owner/repository format.
            This info return from the search_github_repositories function.
    """

    url = f"{GITHUB_API_BASE_URL}/repos/{full_name}"

    response = requests.get(
        url,
        headers=_get_headers(),
        timeout=15,
    )

    response.raise_for_status()

    item = response.json()  # Return the repo details in JSON format

    license_info = item.get("license", {})

    # selectively return only the relevant fields from the repo details.
    return {
        "name": item["name"],
        "full_name": item["full_name"],
        "description": item.get("description"),

        "html_url": item["html_url"],

        "stars": item["stargazers_count"],
        "forks": item["forks_count"],
        "watchers": item["subscribers_count"],
        "open_issues": item["open_issues_count"],

        "language": item.get("language"),

        "topics": item.get("topics", []),

        "license": (
            license_info.get("spdx_id")
            if license_info
            else None
        ),

        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
        "pushed_at": item["pushed_at"],

        "default_branch": item["default_branch"],

        "size_kb": item["size"],

        "archived": item["archived"],
        "fork": item["fork"],
    }


@tool
def get_repository_readme(
        full_name: str, 
        max_char: int = 6400, 
) -> str: 
    """
    Read the README of a GitHub repository.

    Use this tool when you need to understand the repository's purpose,
    architecture, dependencies, setup process, or learning difficulty.

    Args:
        full_name:
            Repository identifier in owner/repository format.
            This info return from the search_github_repositories function.

        max_char:
            Maximum number of characters to return from the README content.
    """

    # first, build the URL to specific content needed.
    url = f"{GITHUB_API_BASE_URL}/repos/{full_name}/readme"

    # second, make a GET request.
    response = requests.get(
        url,
        headers=_get_headers(),
        timeout=15,
    )

    if response.status_code == 404:
        return "README not found."

    response.raise_for_status()

    data = response.json()
    readme_content = data.get("content", "")
    if not readme_content:
        return "README is empty."

    # The README content is base64 encoded, so we need to decode it.
    readme = base64.b64decode(
        readme_content
    ).decode(
        "utf-8",
        errors="replace",
    )

    # Can be better designed to handle large README files.
    return readme[:max_char]


@tool
def get_repository_structure(
        full_name: str, 
        branch: str,
        max_entries: int = 120,
) -> dict: 
    """
    Get the file and directory structure of a GitHub repository.

    Use this tool when you need to inspect the repository's code
    organization, project complexity, or major components.

    Args:
        full_name:
            Repository identifier in owner/repository format.

        branch:
            Repository branch to inspect, usually the default branch.

        max_entries:
            Maximum number of tree entries returned.
    """

    url = (
        f"{GITHUB_API_BASE_URL}/repos/"
        f"{full_name}/git/trees/{branch}"
    )

    response = requests.get(
        url,
        headers=_get_headers(),
        params={
            "recursive": "1",
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    tree = data.get("tree", []) # contain return tree or empty list if not found.

    entries = []
    for item in tree[:max_entries]:

        entries.append(
            {
                "path": item.get("path"),
                "type": item.get("type"),
                "size": item.get("size"),
            }
        )

    return {
        "entries": entries,             # files tree arrangement
        "total_entries": len(tree),     # total number of entries in the tree
        "github_truncated": data.get(
            "truncated",
            False,
        ),
        "reposcout_truncated": (
            len(tree) > max_entries
        ),
    }


# if __name__ == "__main__": 

#     result = search_github_repositories(
#         query = "langgraph agent language:python",
#         limit = 5
#     )

#     # for repo in result: 
#     #     print(json.dumps(repo, indent=4))

#     best_name = result[0].get("full_name")

#     repo_info = get_repository_info(best_name)

#     print(
#         json.dumps(repo_info, indent=4)
#     )

#     readme_content = get_repository_readme(best_name)

#     print(readme_content)


if __name__ == "__main__":

    result = search_github_repositories.invoke(
        {
            "query": "langgraph agent language:python",
            "limit": 5,
        }
    )

    best_name = result[0]["full_name"]

    repo_info = get_repository_info.invoke(
        {
            "full_name": best_name
        }
    )

    print(
        json.dumps(
            repo_info,
            indent=4,
        )
    )

    readme_content = get_repository_readme.invoke(
        {
            "full_name": best_name
        }
    )

    print(readme_content)
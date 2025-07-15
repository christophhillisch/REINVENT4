"""Synthetic feasibility scoring component using external API"""

from __future__ import annotations

__all__ = ["SyntheticFeasibility"]
from typing import List
import requests
import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the synthetic feasibility scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    server_url: List[str] = None
    server_port: List[int] = None
    server_endpoint: List[str] = None


DEFAULT_SERVER_URL = "http://localhost"
DEFAULT_SERVER_PORT = 5000
DEFAULT_SERVER_ENDPOINT = "synthetic_feasibility_surrogate"


@add_tag("__component")
class SyntheticFeasibility:
    def __init__(self, params: Parameters):
        self.server_urls = params.server_url or [DEFAULT_SERVER_URL]
        self.server_ports = params.server_port or [DEFAULT_SERVER_PORT]
        self.server_endpoints = params.server_endpoint or [DEFAULT_SERVER_ENDPOINT]
        
        # needed in the normalize_smiles decorator
        self.smiles_type = "rdkit_smiles"

        self.number_of_endpoints = len(self.server_urls)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> ComponentResults:
        scores = []

        for url, port, endpoint in zip(
            self.server_urls,
            self.server_ports,
            self.server_endpoints,
        ):
            full_url = f"{url}:{port}/{endpoint}"
            json_data = {"smiles": smilies}
            
            try:
                response = requests.post(
                    full_url, 
                    json=json_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise ValueError(
                        f"Synthetic feasibility API failed.\n"
                        f"Status Code: {response.status_code}\n"
                        f"Reason: ({response.reason})\n"
                        f"Response content: {response.content}\n"
                        f"Response text: {response.text}"
                    )
                
                response_json = response.json()
                results = self._parse_response(response_json, len(smilies))
                scores.append(results)
                
            except requests.exceptions.RequestException:
                # If the request fails, return NaN for all molecules
                results = np.full(len(smilies), np.nan, dtype=np.float32)
                scores.append(results)

        return ComponentResults(scores)

    def _parse_response(self, response_json: List[dict], data_size: int) -> np.ndarray:
        """Parse the response from the synthetic feasibility API
        
        Expected response format:
        [
            {"smiles": "C1=CC=CC=C1", "prediction": 0.5760770598287769},
            {"smiles": "CCO", "prediction": 0.5720232508128327},
            ...
        ]
        """
        results = np.full(data_size, np.nan, dtype=np.float32)
        
        # The response should be a list of dictionaries
        if not isinstance(response_json, list):
            return results
            
        # Fill in the results based on the response order
        # The API should return results in the same order as the input
        for i, result in enumerate(response_json):
            if i >= data_size:
                break
            try:
                prediction = result.get("prediction")
                if prediction is not None:
                    results[i] = float(prediction)
            except (ValueError, TypeError, KeyError):
                pass  # Keep NaN for failed predictions
        
        return results

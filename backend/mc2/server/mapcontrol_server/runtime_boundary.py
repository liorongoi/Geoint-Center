"""Capabilities of standalone MC² that are disabled inside Geo Intelligence."""
import os


def embedded():
    return bool(os.getenv('SPECTRASCOPE_BROWSER_ORIGINS') or
                os.getenv('SPECTRASCOPE_NETWORK_MODE') or
                os.getenv('SPECTRASCOPE_OFFLINE') == '1')

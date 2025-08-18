import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
print(settings.DATA_source)
print(settings.COPPER_PRICE_DEF)
print(settings.ZINC_PRICE_DEF)

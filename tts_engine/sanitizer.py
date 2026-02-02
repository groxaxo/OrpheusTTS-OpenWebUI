"""
Text Sanitization Module for Orpheus TTS

This module provides comprehensive text sanitization and normalization
based on strategies from Kokoro-FastAPI. It ensures input text is properly
processed for high-quality speech synthesis.

Features:
- URL normalization
- Email normalization  
- Number and money normalization
- Time and phone number normalization
- Unit normalization
- Symbol replacement
- CJK punctuation handling
- Whitespace normalization
"""

import re
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationOptions:
    """Configuration options for text normalization."""
    
    normalize: bool = True  # Master switch for all normalization
    url_normalization: bool = True
    email_normalization: bool = True
    number_normalization: bool = True
    money_normalization: bool = True
    unit_normalization: bool = False  # Off by default, can be verbose
    time_normalization: bool = True
    phone_normalization: bool = True
    optional_pluralization_normalization: bool = True
    replace_remaining_symbols: bool = True
    expand_abbreviations: bool = True


class TextSanitizer:
    """
    Text sanitization and normalization for TTS input.
    
    Based on strategies from Kokoro-FastAPI TEXT_SANITIZATION_GUIDE.
    """
    
    # Regex patterns
    URL_PATTERN = re.compile(
        r"(https?://|www\.)[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?"
        , re.IGNORECASE
    )
    
    EMAIL_PATTERN = re.compile(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        re.IGNORECASE
    )
    
    MONEY_PATTERN = re.compile(
        r"(-?)([$£€])(\d+(?:\.\d+)?)(k|m|b|t)?(?=\s|$|\.|,)",
        re.IGNORECASE
    )
    
    NUMBER_PATTERN = re.compile(
        r"\b(-?)(\d+(?:\.\d+)?)(k|m|b|t)?\b",
        re.IGNORECASE
    )
    
    TIME_PATTERN = re.compile(
        r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s?(am|pm))?\b",
        re.IGNORECASE
    )
    
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,2})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
    )
    
    UNIT_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?)\s*(km|m|cm|mm|mi|ft|kg|g|lb|oz|gb|mb|kb|tb|mph|kph|°c|°f)\b",
        re.IGNORECASE
    )
    
    # Number word mappings
    ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    
    TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    SCALES = ["", "thousand", "million", "billion", "trillion"]
    
    MULTIPLIERS = {
        "k": "thousand",
        "m": "million", 
        "b": "billion",
        "t": "trillion"
    }
    
    MONEY_UNITS = {
        "$": ("dollar", "cent"),
        "£": ("pound", "pence"),
        "€": ("euro", "cent")
    }
    
    VALID_UNITS = {
        # Length
        "m": "meter", "cm": "centimeter", "km": "kilometer", "mm": "millimeter",
        "ft": "foot", "mi": "mile",
        # Mass
        "g": "gram", "kg": "kilogram", "lb": "pound", "oz": "ounce",
        # Data
        "kb": "kilobyte", "mb": "megabyte", "gb": "gigabyte", "tb": "terabyte",
        # Speed
        "mph": "mile per hour", "kph": "kilometer per hour",
        # Temperature
        "°c": "degree celsius", "°f": "degree fahrenheit",
    }
    
    SYMBOL_REPLACEMENTS = {
        '~': ' ',
        '@': ' at ',
        '#': ' number ',
        '$': ' dollar ',
        '%': ' percent ',
        '^': ' ',
        '&': ' and ',
        '*': ' ',
        '_': ' ',
        '|': ' ',
        '\\': ' ',
        '/': ' slash ',
        '=': ' equals ',
        '+': ' plus ',
    }
    
    def __init__(self, options: Optional[NormalizationOptions] = None):
        """Initialize the sanitizer with given options."""
        self.options = options or NormalizationOptions()
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize and normalize text for TTS processing.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized and normalized text
        """
        if not self.options.normalize:
            return text
        
        if not text or not text.strip():
            return text
            
        logger.debug(f"Sanitizing text: {text[:100]}...")
        
        # Apply normalization steps in order
        if self.options.url_normalization:
            text = self._normalize_urls(text)
            
        if self.options.email_normalization:
            text = self._normalize_emails(text)
            
        if self.options.phone_normalization:
            text = self._normalize_phone_numbers(text)
            
        if self.options.time_normalization:
            text = self._normalize_time(text)
            
        if self.options.money_normalization:
            text = self._normalize_money(text)
            
        if self.options.unit_normalization:
            text = self._normalize_units(text)
            
        if self.options.number_normalization:
            text = self._normalize_numbers(text)
            
        if self.options.expand_abbreviations:
            text = self._expand_abbreviations(text)
            
        if self.options.optional_pluralization_normalization:
            text = self._normalize_optional_pluralization(text)
            
        # Character sanitization
        text = self._normalize_quotes(text)
        text = self._normalize_cjk_punctuation(text)
        
        if self.options.replace_remaining_symbols:
            text = self._replace_symbols(text)
            
        text = self._normalize_whitespace(text)
        
        logger.debug(f"Sanitized result: {text[:100]}...")
        return text
    
    def _normalize_urls(self, text: str) -> str:
        """Convert URLs to pronounceable format."""
        def handle_url(match):
            url = match.group(0).strip()
            
            # Handle protocol
            if url.startswith("https://"):
                url = "https " + url[8:]
            elif url.startswith("http://"):
                url = "http " + url[7:]
            elif url.startswith("www."):
                url = "www " + url[4:]
            
            # Replace dots with "dot"
            url = url.replace(".", " dot ")
            
            # Handle common URL components
            url = url.replace("/", " slash ")
            url = url.replace("?", " question-mark ")
            url = url.replace("=", " equals ")
            url = url.replace("&", " and ")
            url = url.replace(":", " colon ")
            
            # Clean up extra spaces
            url = re.sub(r'\s+', ' ', url)
            
            return url.strip()
        
        return self.URL_PATTERN.sub(handle_url, text)
    
    def _normalize_emails(self, text: str) -> str:
        """Convert email addresses to speakable format."""
        def handle_email(match):
            email = match.group(0)
            user, domain = email.split("@")
            
            # Handle dots in username
            user = user.replace(".", " dot ")
            
            # Handle domain dots
            domain = domain.replace(".", " dot ")
            
            return f"{user} at {domain}"
        
        return self.EMAIL_PATTERN.sub(handle_email, text)
    
    def _normalize_phone_numbers(self, text: str) -> str:
        """Convert phone numbers to spoken digit groups."""
        def handle_phone(match):
            phone = match.group(0)
            
            # Remove formatting, keep only digits and +
            digits = re.sub(r'[^\d+]', '', phone)
            
            # Handle country code
            if digits.startswith('+'):
                digits = 'plus ' + digits[1:]
            
            # Group digits for natural speech
            # Speak each digit individually
            result = ' '.join(digits.replace('+', 'plus'))
            
            return result
        
        return self.PHONE_PATTERN.sub(handle_phone, text)
    
    def _normalize_time(self, text: str) -> str:
        """Convert time expressions to spoken format."""
        def handle_time(match):
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = match.group(3)
            period = match.group(4)
            
            result = [self._number_to_words(hours)]
            
            if minutes == 0:
                result.append("o'clock")
            elif minutes < 10:
                result.append(f"oh {self._number_to_words(minutes)}")
            else:
                result.append(self._number_to_words(minutes))
            
            if seconds:
                result.append(f"and {self._number_to_words(int(seconds))} seconds")
            
            if period:
                result.append(period.lower())
            
            return " ".join(result)
        
        return self.TIME_PATTERN.sub(handle_time, text)
    
    def _normalize_money(self, text: str) -> str:
        """Convert monetary values to spoken format."""
        def handle_money(match):
            negative = match.group(1) == "-"
            currency = match.group(2)
            amount = float(match.group(3))
            multiplier = match.group(4)
            
            bill, coin = self.MONEY_UNITS.get(currency, ("unit", "cent"))
            
            if multiplier:
                multiplier_word = self.MULTIPLIERS.get(multiplier.lower(), "")
                amount_word = self._number_to_words(amount)
                unit = self._pluralize(bill, amount)
                result = f"{amount_word} {multiplier_word} {unit}"
            elif amount == int(amount):
                # Whole amount
                amount_word = self._number_to_words(int(amount))
                unit = self._pluralize(bill, int(amount))
                result = f"{amount_word} {unit}"
            else:
                # With cents
                dollars = int(amount)
                cents_str = str(amount).split(".")[-1].ljust(2, "0")[:2]
                cents = int(cents_str)
                
                dollar_word = self._number_to_words(dollars)
                dollar_unit = self._pluralize(bill, dollars)
                cent_word = self._number_to_words(cents)
                cent_unit = self._pluralize(coin, cents)
                
                result = f"{dollar_word} {dollar_unit} and {cent_word} {cent_unit}"
            
            if negative:
                result = f"minus {result}"
            
            return result
        
        return self.MONEY_PATTERN.sub(handle_money, text)
    
    def _normalize_units(self, text: str) -> str:
        """Convert measurements with units to spoken form."""
        def handle_unit(match):
            number = match.group(1)
            unit = match.group(2).lower()
            
            if unit in self.VALID_UNITS:
                unit_word = self.VALID_UNITS[unit]
                
                # Handle pluralization
                try:
                    num = float(number)
                    unit_word = self._pluralize(unit_word, num)
                except ValueError:
                    pass
                
                return f"{number} {unit_word}"
            
            return match.group(0)
        
        return self.UNIT_PATTERN.sub(handle_unit, text)
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert numeric values to written words."""
        def handle_number(match):
            negative = match.group(1) == "-"
            number_str = match.group(2)
            multiplier = match.group(3)
            
            try:
                number = float(number_str)
            except ValueError:
                return match.group(0)
            
            # Special handling for 4-digit years (1500-2100)
            if (number == int(number) and 
                1500 <= number <= 2100 and 
                len(str(int(number))) == 4 and
                not multiplier):
                year = int(number)
                # Split year into two parts (e.g., 1998 -> "nineteen ninety-eight")
                first = year // 100
                second = year % 100
                if second == 0:
                    result = f"{self._number_to_words(first)} hundred"
                elif second < 10:
                    result = f"{self._number_to_words(first)} oh {self._number_to_words(second)}"
                else:
                    result = f"{self._number_to_words(first)} {self._number_to_words(second)}"
            else:
                result = self._number_to_words(number)
            
            if multiplier:
                multiplier_word = self.MULTIPLIERS.get(multiplier.lower(), "")
                result = f"{result} {multiplier_word}"
            
            if negative:
                result = f"minus {result}"
            
            return result
        
        return self.NUMBER_PATTERN.sub(handle_number, text)
    
    def _number_to_words(self, n: float) -> str:
        """Convert a number to words."""
        if n == int(n):
            n = int(n)
        else:
            # Handle decimals
            parts = str(n).split(".")
            integer_part = self._integer_to_words(int(parts[0]))
            decimal_part = " ".join(self._digit_to_word(d) for d in parts[1])
            return f"{integer_part} point {decimal_part}"
        
        return self._integer_to_words(n)
    
    def _integer_to_words(self, n: int) -> str:
        """Convert an integer to words."""
        if n == 0:
            return "zero"
        
        if n < 0:
            return f"minus {self._integer_to_words(-n)}"
        
        if n < 20:
            return self.ONES[n]
        
        if n < 100:
            tens = n // 10
            ones = n % 10
            if ones == 0:
                return self.TENS[tens]
            return f"{self.TENS[tens]}-{self.ONES[ones]}"
        
        if n < 1000:
            hundreds = n // 100
            remainder = n % 100
            if remainder == 0:
                return f"{self.ONES[hundreds]} hundred"
            return f"{self.ONES[hundreds]} hundred and {self._integer_to_words(remainder)}"
        
        # Handle larger numbers with scales
        result = []
        scale_idx = 0
        
        while n > 0:
            if n % 1000 != 0:
                chunk = self._integer_to_words(n % 1000)
                if self.SCALES[scale_idx]:
                    chunk = f"{chunk} {self.SCALES[scale_idx]}"
                result.insert(0, chunk)
            n //= 1000
            scale_idx += 1
        
        return " ".join(result)
    
    def _digit_to_word(self, digit: str) -> str:
        """Convert a single digit to word."""
        digit_map = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
        }
        return digit_map.get(digit, digit)
    
    def _pluralize(self, word: str, count: float) -> str:
        """Add appropriate plural suffix to a word."""
        if count == 1 or count == -1:
            return word
        
        # Handle special cases
        if word == "foot":
            return "feet"
        if word.endswith("y"):
            return word[:-1] + "ies"
        if word.endswith(("s", "sh", "ch", "x", "z")):
            return word + "es"
        
        return word + "s"
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations for better pronunciation."""
        # Titles - only when followed by a capital letter
        text = re.sub(r"\bDr\.(?=\s+[A-Z])", "Doctor", text)
        text = re.sub(r"\bMr\.(?=\s+[A-Z])", "Mister", text)
        text = re.sub(r"\bMs\.(?=\s+[A-Z])", "Ms", text)  # Pronounced "mizz"
        text = re.sub(r"\bMrs\.(?=\s+[A-Z])", "Mrs", text)
        text = re.sub(r"\bProf\.(?=\s+[A-Z])", "Professor", text)
        
        # Common abbreviations
        text = re.sub(r"\betc\.(?!\s+[A-Z])", "etcetera", text)
        text = re.sub(r"\bvs\.?\b", "versus", text)
        text = re.sub(r"\bi\.e\.?\b", "that is", text, flags=re.IGNORECASE)
        text = re.sub(r"\be\.g\.?\b", "for example", text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_optional_pluralization(self, text: str) -> str:
        """Expand optional pluralization markers like (s)."""
        text = re.sub(r"\(s\)", "s", text)
        return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Standardize various quote styles to ASCII quotes."""
        # Unicode quotes to ASCII
        text = text.replace(chr(8216), "'")  # '
        text = text.replace(chr(8217), "'")  # '
        text = text.replace(chr(8220), '"')  # "
        text = text.replace(chr(8221), '"')  # "
        
        # Guillemets to quotes
        text = text.replace("«", '"')
        text = text.replace("»", '"')
        
        return text
    
    def _normalize_cjk_punctuation(self, text: str) -> str:
        """Convert Chinese/Japanese punctuation to Western equivalents."""
        replacements = {
            '、': ', ',   # Enumeration comma
            '。': '. ',   # Period
            '！': '! ',   # Exclamation
            '，': ', ',   # Comma
            '：': ': ',   # Colon
            '；': '; ',   # Semicolon
            '？': '? ',   # Question mark
            '–': '- ',    # En dash
        }
        
        for cjk, western in replacements.items():
            text = text.replace(cjk, western)
        
        return text
    
    def _replace_symbols(self, text: str) -> str:
        """Replace symbols with spoken equivalents."""
        for symbol, replacement in self.SYMBOL_REPLACEMENTS.items():
            # Don't replace $ if already processed as money
            if symbol == '$' and self.options.money_normalization:
                continue
            text = text.replace(symbol, replacement)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Clean and standardize all whitespace."""
        # Remove non-standard whitespace (tabs, etc.)
        text = re.sub(r"[^\S \n]", " ", text)
        
        # Collapse multiple spaces
        text = re.sub(r"  +", " ", text)
        
        # Remove spaces in blank lines
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)
        
        # Convert newlines to spaces
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        return text.strip()

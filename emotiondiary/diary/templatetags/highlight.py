from django import template
import re
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def highlight(text, query):
    if not query:
        return text
    # 정규식으로 검색어를 찾아 <mark> 태그로 감싸기
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    return mark_safe(highlighted)

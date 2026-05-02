from openrouter import OpenRouter
import os

with OpenRouter(
  api_key="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",
) as client:
  response = client.chat.send(
    #model="meta-llama/llama-3.2-1b-instruct",
    model="qwen/qwen-2.5-7b-instruct",
    max_tokens=50,   # 👈 限制長度
    messages=[
      {
        "role": "user",
        "content": "我應該怎麼選擇租屋?用一句話回答我"
      }
    ]
  )

  print(response.choices[0].message.content)
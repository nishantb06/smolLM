# Backend for serving the SmolLM model

## sample curl request

```
curl -X POST "http://localhost:8001/text/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Once upon a time",
           "max_tokens": 100,
           "temperature": 0.8,
           "top_k": 40
         }'

curl http://localhost:8001/health
```

## docker commands

```
docker run -p 8001:8001 -v /Users/nishantbhansali/Desktop/personal/eraV3/smolLM/application/backend:/app/weights --name smollm-backend-container smollm-backend

docker build -t smollm-backend .
```

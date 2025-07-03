Simple Fastapi example to make inference with tts models in onnx format. 

Steps: 

1. Clone the repo.
`git clone https://github.com/langtech-bsc/minimal-tts-api`
2. Build and run the container.
```
cd minimal-tts-api
docker build -t minimal-tts-api .
docker run -p 8000:8000 -t minimal-tts-api
```
3. Test with a simple request.

```
curl -X POST   http://0.0.0.0:8000/api/tts   -H "Content-Type: application/json"   -d '{"text":"Bon dia","voice":"quim","accent":"balear","type":"text"}'   | aplay -t wav -

```

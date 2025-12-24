FROM ollama/ollama
RUN nohup ollama serve & \
    timeout=30; \
    while ! ollama list > /dev/null 2>&1 && [ $timeout -gt 0 ]; do \
        sleep 1; \
        timeout=$((timeout - 1)); \
    done && \
    ollama pull llama3.2:3b
EXPOSE 11434
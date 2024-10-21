CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

SRCS = test-rag.c ./vector-store/hnsw.c ./vector-store/exhaustive.c ./vector-store/document/document.c ./embedding-model/embedding_model.c ./vector-store/priority-queue.c ./vector-store/util.c
OBJS = $(SRCS:.c=.o)
TARGET = test-rag

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)


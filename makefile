OBJS = layer.o model.o main.o
CFLAG = -Wall -g
CC = clang++
INCLUDE = 
LIBS = -lm

main: ${OBJS}
	${CC} ${CFLAG} -o $@ ${OBJS} ${LIBS}

clean:
	rm -f *.o

.c.o:
	${CC} ${CFLAG} -c $<

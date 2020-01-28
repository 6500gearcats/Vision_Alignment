OBJS = layer.o model.o optimize_trained_net.o main.o
CFLAG = -Wall -g
CC = clang++
INCLUDE = 
LIBS = -lm

main.out: ${OBJS}
	${CC} ${CFLAG} -o $@ ${OBJS} ${LIBS}

clean:
	rm -f *.o

.cpp.o:
	${CC} ${CFLAG} -c $<

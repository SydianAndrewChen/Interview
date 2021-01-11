#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define INTERVAL 4e5

char map[ROW+10][COLUMN] ; 

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

struct Log
{
    bool direction; // 0 : l->r , 1: r->l
    int index; // Represent which row this log is on
	int start; 
    int end;
	Log(bool _dir, int _stt, int _end, int _id)
		:direction( _dir ), start(_stt), end( _end ), index( _id ){};
	Log(){};
} * logs;

enum {Gaming, Quit, Fail, Win} state;

pthread_mutex_t map_upd_mutex;
pthread_mutex_t frog_upd_mutex;
pthread_cond_t map_upd_cond;
pthread_attr_t attr;

/* Initalization of map and logs*/
void* logs_init();
void* map_init();

int kbhit(void);

void* frog_move(void* t);

void* logs_move(void* t);
void* log_move(void*t);
int log_index;

int main( int argc, char *argv[] ){
	/* Initialization */
	state = Gaming;
	frog = Node( ROW, (COLUMN-1) / 2 ); 
	logs = new Log[ROW-1];

	pthread_mutex_init(&map_upd_mutex, NULL);
	pthread_mutex_init(&frog_upd_mutex, NULL);
	pthread_cond_init(&map_upd_cond, NULL);

	map_init();
	logs_init(); 


	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	pthread_t tid1, tid2;
	/*  Move the logs  */
	pthread_create(&tid1, &attr, logs_move, NULL);

	/*  Check keyboard hits, to change frog's position or quit the game. */
	pthread_create(&tid2, &attr, frog_move, NULL);

	while (state == Gaming){
		pthread_mutex_trylock(&map_upd_mutex);
			pthread_cond_wait(&map_upd_cond, &map_upd_mutex);
			/*  Print the map on the screen  */
			printf("\033c\033[?25l"); // "\033[?25l" helps to hide the cursor
			for( int i = 0; i <= ROW; i++){
				for(int j = 0; j < COLUMN; j++){
					if ((i == frog.x)&&(j == frog.y)) 
						printf("0");
					else
						printf("%c",map[i][j]);
				}
				printf("\n");
			}
		pthread_mutex_unlock(&map_upd_mutex);
	}	
	pthread_join(tid1, NULL);
	pthread_join(tid2, NULL);

	
	pthread_mutex_destroy(&map_upd_mutex);
	pthread_mutex_destroy(&frog_upd_mutex);
	pthread_cond_destroy(&map_upd_cond);
	delete[] logs;

	pthread_attr_destroy(&attr);

	/* Check game status */
	printf("\033c");
	switch (state)
	{
		case 1:
			printf("Quitted!\n");
			break;
		case 2:
			printf("Failed!\n");
			break;
		case 3:
			printf("Win!\n");
			break;
		default:
			break;
	}
	return 0;

}

/*-------------------------------------------*/
/*                Move Functions             */
/*-------------------------------------------*/
void *log_move(void * t){
	if (!logs[log_index-1].direction) // From left to right
	{
		(++logs[log_index-1].end)   %= (COLUMN-1); // Cyclic order
		(++logs[log_index-1].start) %= (COLUMN-1);
		if ((logs[log_index-1].start-1) < 0) logs[log_index-1].start += (COLUMN-1); // Cpp's mod use floor, thus we need extra operations
		if ((logs[log_index-1].end-1)   < 0) logs[log_index-1].end   += (COLUMN-1);
		map[log_index][logs[log_index-1].end-1]   = '=';
		map[log_index][logs[log_index-1].start-1] = ' ';
	}
	else // From right to left
	{
		(--logs[log_index-1].end)   %= (COLUMN-1); // Cyclic order
		(--logs[log_index-1].start) %= (COLUMN-1);
		if (logs[log_index-1].start < 0) logs[log_index-1].start += (COLUMN-1); // Cpp's mod use floor, thus we need extra operations
		if (logs[log_index-1].end   < 0) logs[log_index-1].end   += (COLUMN-1);
		map[log_index][logs[log_index-1].start] = '=';
		map[log_index][logs[log_index-1].end]   = ' ';
	}
	pthread_exit(NULL);
	return 0;
}

void *logs_move( void *t ){
	pthread_t tid;
	usleep(INTERVAL);
	pthread_create(&tid, &attr, logs_move, NULL);

	pthread_mutex_lock(&map_upd_mutex);
	pthread_t* tids = new pthread_t[ROW-1];
		for (int i = 1; i < ROW;i++){
			log_index = i;
			pthread_create(&tids[log_index-1], NULL, log_move, &log_index);
			pthread_join(tids[log_index-1], NULL);
		}
		pthread_cond_signal(&map_upd_cond);

		pthread_mutex_trylock(&frog_upd_mutex);
			if ((frog.x != 0) && (frog.x != ROW))
				(logs[frog.x-1].direction)? frog.y--:frog.y++; // 0 : l->r
			if (frog.y < 0 || frog.y > COLUMN-1) state = Fail;
		pthread_mutex_unlock(&frog_upd_mutex);
	pthread_mutex_unlock(&map_upd_mutex);
	pthread_exit(NULL);
	return 0;	
}

void* frog_move(void* t){
	while(state == Gaming){
		while (!kbhit()) ; // Hangup
			pthread_mutex_trylock(&frog_upd_mutex);
				char dir = getchar();
				if( dir == 'q' || dir == 'Q' ){
					state = Quit;
				}
				if( dir == 'w' || dir == 'W' )
					frog.x--;
				else if( dir == 'a' || dir == 'A' )
					frog.y--;
				else if( dir == 'd' || dir == 'D' )	 
					frog.y++;
				else if( dir == 's' || dir == 'S' )
					if (frog.x <= ROW-1) frog.x++;

				if (map[frog.x][frog.y] == ' ') state = Fail;
				if (frog.x == 0) state = Win;
			pthread_mutex_unlock(&frog_upd_mutex);
			pthread_cond_signal(&map_upd_cond);
	}
	pthread_exit(NULL);
	return 0;
}

/*-------------------------------------------*/
/*                Init Functions             */
/*-------------------------------------------*/
void * map_init(){
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	return 0;
}

void * logs_init(){
	srand((unsigned)time(NULL));
	int temp, start, end, length;
	printf("\nLoading.");
	for (int i = 0; i < ROW-1; i++){
		temp = rand()%100;
		length = temp%(COLUMN/4) + COLUMN/8;
		start = ((i%2)*(COLUMN - length - 1) + temp) %(COLUMN-1); // 	`temp%2` stands for the direction of this log, if temp%2 = 0 (false), 
												// then log will start from left side, floating towards right.
		end   = start + length;
		for (int j = start; j < end; j++){
			map[i+1][j%(COLUMN-1)] = '=';
		}
		Log log = Log(i%2, start, end, i);
		logs[i] = log;
		printf(".");
	}
	return 0;
}

/*-------------------------------------------*/
/*           Keyboad Check Functions         */
/*-------------------------------------------*/
// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}
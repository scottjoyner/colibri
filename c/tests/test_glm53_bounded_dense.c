/* Fork-only bounded dense conversion test: numerical equivalence and rejection.
 * Real safetensors indexing is tested separately in glm53 fixture harnesses.
 */
#define main glm53_hidden_main
#include "../glm53.c"
#undef main
#include <sys/wait.h>

static int equal(const Mat *a, const Mat *b, int rows, int cols) {
    if (a->fmt != b->fmt || a->rows != b->rows || a->columns != b->columns) return 0;
    if (a->fmt == 4) {
        const size_t bytes = (size_t)rows * cols / 2;
        const size_t scale_bytes = (size_t)rows * (cols / 64) * sizeof(float);
        return memcmp(a->q4,b->q4,bytes)==0 && memcmp(a->s,b->s,scale_bytes)==0;
    }
    return memcmp(a->q8,b->q8,(size_t)rows*cols)==0
        && memcmp(a->s,b->s,(size_t)rows*sizeof(float))==0;
}
static void release(Mat *mat) {
    glm53_matrix_host_release(mat);
}
static int test_case(int rows,int cols,int dtype,int stage) {
    const size_t N=(size_t)rows*cols, esz=st_dtype_esz(dtype);
    char name[]="/tmp/glm53-bounded-XXXXXX";
    int fd=mkstemp(name);
    if (fd<0){perror("mkstemp");return 1;}
    unlink(name);
    const int64_t off=512;
    char zeros[512]={0};
    if (write(fd,zeros,sizeof zeros)!=sizeof zeros)abort();
    uint8_t *raw=malloc(N*esz);
    float *oracle=malloc(N*sizeof(float));
    if(!raw||!oracle)abort();
    for(size_t i=0;i<N;i++){
        const float v=(float)((int)((i*17+i/cols*13)%191)-95)/91.f;
        if(dtype==2) {
            memcpy(raw+i*4,&v,4);
            oracle[i]=v;
        } else if(dtype==0) {
            uint32_t b;memcpy(&b,&v,4);
            uint16_t w=(uint16_t)(b>>16); memcpy(raw+i*2,&w,2);
        } else {
            /* Positive/negative, subnormal and largest finite f16 inputs. */
            const uint16_t words[]={0x3c00,0xbc00,0x3555,0x0001,0x7bff,0x0400,0x0000,0x8000};
            uint16_t w=words[(i+i/cols)%8];memcpy(raw+i*2,&w,2);
        }
    }
    if (dtype==0) bf16_to_f32_bulk((uint16_t*)raw,oracle,N);
    if (dtype==1) f16_to_f32_bulk((uint16_t*)raw,oracle,N);
    if(pwrite(fd,raw,N*esz,off)!=(ssize_t)(N*esz))abort();
    st_tensor t={.fd=fd,.off=off,.nbytes=(int64_t)(N*esz),.numel=(int64_t)N,
                 .rank=2,.dtype=dtype,.shape={rows,cols}};
    if(stage==1)t.nbytes-=esz;
    if(stage==2)t.dtype=3;
    if(stage==3 && ftruncate(fd,off+N*esz-16))abort();
    if(stage) {
        pid_t child=fork();
        if(child==0){Mat bad=glm53_bounded_quant_tensor(&t,rows,cols);release(&bad);_exit(0);}
        int status;waitpid(child,&status,0);
        close(fd);free(raw);free(oracle);
        if(WIFEXITED(status) && WEXITSTATUS(status)!=0) return 0;
        if(WIFSIGNALED(status))return 0;
        fprintf(stderr,"FAIL: invalid tensor stage %d accepted\n",stage);
        return 1;
    }
    Mat expected=quantize_loaded(oracle,rows,cols);
    Mat received=glm53_bounded_quant_tensor(&t,rows,cols);
    int ok=equal(&expected,&received,rows,cols);
    if(!ok)fprintf(stderr,"FAIL: numerical mismatch %d x %d dtype %d\n",rows,cols,dtype);
    release(&expected);release(&received);close(fd);free(raw);
    return !ok;
}
int main(void) {
    int fail=0;
    const char *stored=getenv("GLM53_DENSE_SPILL_DIR");
    char template[]="/tmp/glm53-dense-unit-XXXXXX";
    char *temp=mkdtemp(template);
    if(!temp){perror("mkdtemp");return 1;}
    for(int spill=0;spill<2;spill++){
        if(spill)setenv("GLM53_DENSE_SPILL_DIR",temp,1);
        else unsetenv("GLM53_DENSE_SPILL_DIR");
        /* More than one bounded row window, gs64 and fallback-int8. */
        for(int dtype=0;dtype<=2;dtype++){
            fail+=test_case(96,8192,dtype,0);
            fail+=test_case(81,65,dtype,0);
            fail+=test_case(19,128,dtype,0);
        }
        fail+=test_case(16,128,0,1); /* header byte count mismatch */
        fail+=test_case(16,128,0,2); /* wrong dtype */
        fail+=test_case(16,128,0,3); /* short underlying file */
    }
    DIR *d=opendir(temp);
    if(!d)abort();
    struct dirent *entry;
    while((entry=readdir(d)))if(strcmp(entry->d_name,".") && strcmp(entry->d_name,"..")){
        fprintf(stderr,"FAIL: spill artifact left behind: %s\n",entry->d_name);fail++;
    }
    closedir(d);rmdir(temp);
    if(stored)setenv("GLM53_DENSE_SPILL_DIR",stored,1);
    else unsetenv("GLM53_DENSE_SPILL_DIR");
    if(fail){fprintf(stderr,"FAIL bounded GLM53 conversion: %d cases\n",fail);return 1;}
    puts("PASS 24 bounded dense tests: 18 numerical equivalents, 6 hostile/truncated rejections, spill cleanup");
    return 0;
}

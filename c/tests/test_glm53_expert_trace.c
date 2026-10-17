/* Opt-in capture must preserve model math, bounded file ownership, and privacy. */
#define main glm53_unused_main
#include "../glm53.c"
#undef main
#include <sys/wait.h>

int main(void) {
    char root[]="/tmp/joyner-g53-trace-XXXXXX";
    if(!mkdtemp(root))return 1;
    char path[512];snprintf(path,sizeof path,"%s/trace.jsonl",root);
    GModel model={0};
    model.c.hidden=32;
    model.c.n_experts=288;
    model.c.n_layers=45;
    model.ecache=calloc(45,sizeof(LCache));
    model.ecache[3].n=1;model.ecache[3].cap=2;
    model.ecache[3].s=calloc(2,sizeof(Slot));
    model.ecache[3].s[0].eid=7;
    float features[32],gates[8];
    int chosen[8]={7,8,9,10,11,12,13,14};
    for(int i=0;i<32;i++)features[i]=(float)(i-16)/8;
    for(int i=0;i<8;i++)gates[i]=.125f;
    if(setenv("GLM53_EXPERT_TRACE",path,1))return 2;
    g53_expert_trace_route(&model,3,0,features,chosen,gates,8);
    g53_expert_trace_close();
    struct stat st;
    if(stat(path,&st)|| (st.st_mode&077) || st.st_size<200){
        fprintf(stderr,"FAIL trace permissions or length\n");return 3;
    }
    FILE *f=fopen(path,"r");
    if(!f)return 4;
    char buf[4096];
    if(!fgets(buf,sizeof buf,f))return 5;
    fclose(f);
    if(!strstr(buf,"\"v\":2") ||
       !strstr(buf,"\"ids\":[7,8,9,10,11,12,13,14]") ||
       !strstr(buf,"\"resident\":[1,0,0,0,0,0,0,0]") ||
       !strstr(buf,"\"host_mask\":\"80") ||
       !strstr(buf,"\"gpu_mask\":\"0000") ||
       !strstr(buf,"\"sketch\"") || strstr(buf,"prompt")){
        fputs("FAIL missing data or prompt leak\n",stderr);return 6;
    }
    /* Attempting to reuse the same evidence path must refuse, never overwrite. */
    pid_t p=fork();
    if(p==0) {
        g53_trace_fp=NULL;g53_trace_init_done=0;
        g53_expert_trace_route(&model,3,0,features,chosen,gates,8);
        _exit(0);
    }
    int status;waitpid(p,&status,0);
    if(!WIFEXITED(status)||WEXITSTATUS(status)==0){
        fputs("FAIL existing trace was not refused\n",stderr);return 7;
    }
    /* A completely non-streaming engine has no LCache array at all. It must
     * still be traceable without touching the native resident Mat experts. */
    free(model.ecache[3].s);free(model.ecache);
    model.ecache=NULL;
    char second[512];snprintf(second,sizeof second,"%s/nonstream.jsonl",root);
    g53_trace_fp=NULL;g53_trace_init_done=0;
    setenv("GLM53_EXPERT_TRACE",second,1);
    g53_expert_trace_route(&model,3,0,features,chosen,gates,8);
    g53_expert_trace_close();
    FILE *f2=fopen(second,"r");
    if(!f2 || !fgets(buf,sizeof buf,f2))return 8;
    fclose(f2);
    if(!strstr(buf,"\"host_capacity\":0") ||
       !strstr(buf,"\"resident\":[0,0,0,0,0,0,0,0]")){
        fputs("FAIL nonstreamed cache null protection\n",stderr);return 9;
    }
    unlink(second);
    unlink(path);rmdir(root);
    puts("PASS expert trace: known residency, minimal sketch, 0600, and exclusive-create denial");
    return 0;
}

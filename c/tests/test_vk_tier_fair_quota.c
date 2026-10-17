#include "../vk_tier_fair_quota.h"
#include <stdio.h>
#include <stdint.h>
#define CHECK(c) do{if(!(c)){fprintf(stderr,"NO_GO fair quota line %d\n",__LINE__);return 2;}}while(0)
int main(void){
 for(int cap=0;cap<=256;cap++){
  for(int epoch=1;epoch<=20;epoch++){
   int sum=0,min=10000,max=-1;
   for(int l=3;l<45;l++){
    int n=vkt_fair_quota(cap,3,42,l,epoch);
    sum+=n;if(n<min)min=n;if(n>max)max=n;
   }
   CHECK(sum==cap);
   CHECK(max-min<=1);
  }
 }
 CHECK(!vkt_fair_quota(32,3,42,2,1));
 CHECK(!vkt_fair_quota(32,3,42,45,1));
 CHECK(!vkt_fair_quota(32,3,42,3,0));
 CHECK(!vkt_fair_quota(32,3,0,3,1));
 CHECK(vkt_fair_quota(96,3,42,3,1)>=2);
 int served[42]={0};
 for(int epoch=1;epoch<=42;epoch++)for(int l=3;l<45;l++)
  served[l-3]+=vkt_fair_quota(32,3,42,l,(uint64_t)epoch);
 for(int i=1;i<42;i++)CHECK(served[i]==served[0]);
 puts("PASS fair-share total global promotion cap, per-epoch balance, 42-epoch rotation");
 return 0;
}

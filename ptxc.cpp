#include "nvrtc_wrapper.h"

struct param{
    vector<string> opts;
    string input;
    string output;
    bool is_valid;
};

void print_usage(){
    printf("-------------------------------------------------------------------------------\n");
    printf("usage : ./ptxc -i<input_file> -o<output_file> [options...]  \n");
    printf("<input_file>   : specify single input CUDA source file.\n");
    printf("                 multiple CUDA source files are not supported now ...\n");
    printf("                 maybe it will be supported in future releases :) ... \n");
    printf("<output_file>  : specify output ptx assembly file location.\n");
    printf("[options...] : add compile option. e.g: -arch=compute_61\n");
    printf("A full example is shown here:\n");
    printf("./ptxc  -i./kernel.cu  -o./kernel.ptx  \\   <- compile kernel.cu into kernel.ptx\n");
    printf("          -arch=compute_61             \\   <- with target GPU architecture 6.1 \n");
    printf("          --std=c++14                  \\   <- and C++14 language standard      \n");
    printf("          --use_fast_math                   <- also enables fast math optimizations\n");
    printf("Note that you shouldn't add spaces (' ') beside '='.\n");
    printf("-------------------------------------------------------------------------------\n");
}

param parse_argv(int argc,char** argv)
{
    param p;
    p.is_valid = true;
    printf("args:\n");
    for(int i=1; i<argc; i++){
        string arg = argv[i];
        printf("%s\n",arg.c_str());
        if(arg.size() > 0 && arg[0] == '-'){
            if(arg.size()>=2){
                string t;
                switch(arg[1]){
                    case 'i':
                        t = arg.substr(2);
                        p.input=t;
                    break;
                    case 'o':
                        t = arg.substr(2);
                        p.output=t;
                    break;
                    default:
                        p.opts.push_back(arg);
                }
            }
        }
        else{
            p.opts.push_back(arg);
        }
    }
    return p;
}



void code_dump(const char* code)
{
    unsigned int len=strlen(code);
    printf("code dump (%u chars):\n",len);
    printf("----------------------------------------------------------------------\n");
    int linenum=1;
    printf("%4d |",linenum);
    for(unsigned int i=0;i<len;i++){
        char ch = code[i];
        if (ch=='\n'){
            linenum++;
            printf("\n%4d |", linenum);
            continue;
        }
        if (ch=='\t'){
            printf("    ");
            continue;
        }
        printf("%c",ch);
    }
    printf("\n----------------------------------------------------------------------\n");


}

int main(int argc,char** argv)
{
    if(argc < 2){
        print_usage();
        return 0;
    }
    param p = parse_argv(argc,argv);
    printf("input file : \n%s\n",p.input.c_str());
    printf("output file : \n%s\n",p.output.c_str());
    cuModuleCompiler module;
    if(module.LoadSource(p.input.c_str()) < 0){
        printf("error, cannot load source.\n");
        exit(-1);
    }
    for(size_t i=0;i<p.opts.size();i++){
        module.AddCompileOption(p.opts[i].c_str());
        printf("adding compile option : %s\n", p.opts[i].c_str());
    }

    if (module.Compile() == -1){
        cout << module.GetCompileLog() << endl;
        exit(1);
    }
    else{
        cout<<"Compile success!"<<endl;
    }
    string ptx = module.GetPTX();
    printf("%d bytes.\n",int(ptx.size()));
	//code_dump(ptx.c_str());
    // write to file
    FILE* fp = fopen(p.output.c_str(),"w");
    if(fp == nullptr){
        cout<<"error, cannot open "<<p.output.c_str()<<" for writing."<<endl;
        exit(1);
    }
    else{
        if(fwrite(ptx.c_str(),1,strlen(ptx.c_str()),fp)!=strlen(ptx.c_str())){
            cout<<"error, file writing error.."<<endl;
        }
        else{
            cout<<"PTX code successfully generated!"<<endl;
        }
        fclose(fp);
    }
    return 0;
}

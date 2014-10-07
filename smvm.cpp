#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <vector>
#include <algorithm>
typedef std::pair<int,double> mypair;
//Compressed Column Storage format
class SparseMatrix
{
public:
  typedef std::vector<double> val_vec;
  typedef std::vector<int> ptr_vec;
  val_vec val;
  ptr_vec col_ptr;
  ptr_vec row_ptr;
  int row;
  int col;
  int nnz;

  SparseMatrix(int i){
    row = i; col = i;
  }

  SparseMatrix(int i, int j){
    row = i;
    col = j;
  }

  val_vec multiply(val_vec &x, int nrt)
  {
    if(x.size()!= col)
      throw;
    val_vec result(col,0);
    //Iterate over row pointer vector
    omp_set_num_threads(nrt);
    #pragma omp parallel for shared(result)schedule(dynamic)
    for(int i=0; i < col; i++)
      {
	//std::cout<<"Col num i "<<i<<std::endl;
	//result[i]=0;
	for(int c_index = col_ptr[i]; c_index<col_ptr[i+1]; c_index++)
	  {
	    // std::cout<<"C Index "<<c_index<<std::endl;
	    int j = row_ptr[c_index];
	    //std::cout<<"Row num j "<<j<<std::endl;
	    //std::cout<<"Val "<<val[j]<<std::endl;
	    //std::cout<<"Vector val "<<x[i]<<std::endl;
	    //std::cout<<"Result "<<result[j]<<std::endl;
	    #pragma omp atomic update
	    result[j]+=val[j]*x[i];
	  }
      }
    return result;
  }
  
};

bool comparator (const mypair& l, const mypair& r)
{
  return l.second > r.second;
}

int main(int argc, char* argv[])
{
  if (argc !=3)
    {
      std::cout<<"Invalid input"<<std::endl;
      return 1;
    }

  int nrt = atoi(argv[1]);
  int num_rows = 1770961;
  int num_cols = 1770961;

  int nnz = 83663478;

  std::string input_mat (argv[2]);
  std::ifstream input(input_mat.c_str());
  if (!input)
    {
      std::cout<<"Failed to open file "<<input_mat<<std::endl;
    }

  std::cout<<"Opened Input Matrix File"<<std::endl;

  std::cout<<num_rows<<" "<<num_cols<<std::endl;
  std::cout<<"NNZ "<<nnz<<std::endl;
  SparseMatrix A(num_rows);
  int nnz_count=0;
  int prev_col = -1;
  std::string row,col;
  int r,c;
  int val;
  int counter=0;
  //A.col_ptr.push_back(0);
  for (std::string line;getline(input>>row>>col>>val,line);)
    {
      //std::cout<<atoi(row.c_str())<<" "<<atoi(col.c_str())<<" "<<val<<std::endl;
      r = atoi(row.c_str());
      c = atoi(col.c_str());
      //std::cout<<r<<" "<<c<<std::endl;
      A.val.push_back(val);
      A.row_ptr.push_back(r-1);
      if ((c-1)!= prev_col)
	{
	  int col_size = A.row_ptr.size()-1;
	  for (int k = prev_col; k < (c-1); k++)
	  {
	      //std::cout<<"Prev col "<<prev_col<<"New col "<<c-1<<std::endl;
	      A.col_ptr.push_back(col_size);
	      //std::cout<<counter<<std::endl;
	      counter++;
	   }
	  prev_col = c-1;
	}
      nnz_count++;
    }
  std::cout<<"NNZ "<<nnz_count<<std::endl;
  std::cout<<" Values size "<<A.val.size()<<std::endl;
  std::cout<<" Row ptr size "<<A.row_ptr.size()<<std::endl;
  std::cout<<" Col ptr size "<<A.col_ptr.size()<<std::endl;
  std::cout<<" Counter "<<counter<<std::endl;
  std::cout<<"Finished reading train file"<<std::endl;

  std::vector <double> x;
  srand((unsigned)time(0));
  int random;
  int low = 0, high = 1;
  int range = (high-low)+1;
  for (int index = 0; index < A.col_ptr.size(); index++)
    {
      random = (rand()% range)+ low/(RAND_MAX+1.0);
      x.push_back(random);
      //std::cout<<random<<std::endl;
    }
  std::cout<<x.size()<<std::endl;
  
  std::vector<double> result;
  double start = omp_get_wtime();
  result = A.multiply(x,nrt);
  std::cout<<"Multiply in "<<omp_get_wtime()-start<<" sec"<<std::endl;
  std::cout<<"Result size "<<result.size()<<std::endl;

    
  std::vector<double> y(num_cols,1);
  int T = 50;
  double norm = 0.0;
  start = omp_get_wtime();
  for (int iter = 0; iter<=T; iter++)
    {
      double norm_sq= 0.0;
      
      for (int k=0; k<y.size();k++)
	norm_sq += y[k]*y[k];
      norm = sqrt(norm_sq);
      //std::cout<<"Iteration "<<iter<<" Norm "<<norm<<std::endl;
      norm = 1/norm;
      for(int k = 0; k<y.size();k++)
	{
	  
     
	y[k]*=norm;
	}
      if(iter==T)
	break;
      y=A.multiply(y,nrt);
    }

  std::cout<<"Wall time for Eigen vector centrality is"<<omp_get_wtime()-start<<std::endl;
	
  std::cout<<"Top Eigen value is "<<1/norm<<std::endl;

  //typedef std::pair<int,double> mypair;
  std::vector<mypair> sorted_y;

  for(int i=0; i<y.size();i++)
    sorted_y.push_back(std::make_pair(i,y[i]));

  std::sort(sorted_y.begin(),sorted_y.end(),comparator);

  // std::cout<<"Printing sorted nodes by index and rank "<<std::endl;
  for (int i = 0;  i<sorted_y.size()&& i<100; i++)
    //std::cout<<i+1<<" "<<sorted_y[i].first<<std::endl;
  
  return 0;
}

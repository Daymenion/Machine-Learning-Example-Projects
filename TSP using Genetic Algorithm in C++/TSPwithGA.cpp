//
// Created by Daymenion on 17.04.2022.
//
//TSP solving with generic Algorithm by Mehmet ÜNLÜ
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License

#include<bits/stdc++.h>
#include<cmath>
#include <random>
#include <vector>
#include <map>
#include <set>
#include<ctime>
#include <utility>
#include<cstdlib>
#include <algorithm>
#include<iostream>
#include<fstream>
using namespace std;

int real_size_population = 0; //size of population
int generations = 1000; //number of generations to run default is 1000
int	mutation_rate = 15; //mutation rate in percent
bool show_population_in_terminal = false; //its very large so it is not recommended to use it
bool save_population_in_file = false; //its very large so it is not recommended to use it
int originalGraph[10000][10000];  //original graph maximum size is 10000x10000
int points[10000][2],total_cities;
vector<pair<vector<int>, int> >population;


int generateRandomNumber(int upperbound) {  //generate random number between 0 and upperbound
    random_device rd; //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_int_distribution<> dis(0, upperbound - 1);
    return dis(gen); //return random number
}

int calculateDistance(int x1,int y1,int x2,int y2) //calculate distance between two points using pythagoras theorem (a^2 + b^2 = c^2) and return the result as an integer
{
    return int (sqrt(pow(x1-x2,2)+pow(y1-y2,2))); //rounding the result to an integer value (rounding to the nearest integer) and return the result
}

void get() //get the inputs from the file or generate random inputs for the program
{
    s:
    cout<<"Are you going to give the city locations to the program with txt or do you want them to be assigned randomly?\n";
    cout<<"1.txt\n2.random\n3.exit\n";
    int choice; //choice to give the inputs or not
    cin>>choice;
    cout<<"How many generations do you want to use for the algorithm? (default 1000)\n";
    //if user give 0 or negative number, it will be set to 1000 by default (1000 generations) and the program will run
    cin>>generations;
    if(generations<=0)
        generations=1000;
    // ask the user show the population in terminal and ask the show population in file or not
    cout<<"Do you want to show the population in terminal? It is not recommended to say yes (default no)\n";
    cout<<"1.yes\n2.no\n";
    int choice2;
    cin>>choice2;
    if(choice2==1)
        show_population_in_terminal=true;

    cout<<"Do you want to save the population in file? (default no)\n";
    cout<<"1.yes\n2.no\n";
    cin>>choice2;
    if(choice2==1)
        save_population_in_file=true;

    if(choice == 1) //if the user wants to give the inputs
    {
        cout<<"Enter the name of the file: ";
        char filename[100];
        cin>>filename;
        ifstream fin;
        fin.open(filename);
        if(!fin)
        {
            cout<<"File not found! Try again\n";
            goto s;
        }
        int i = 0;
        while(!fin.eof())
        {
            fin>>points[i][0]>>points[i][1];
            i++;
        }
        total_cities = i;
        fin.close();
    }
    else if(choice == 2) //if the user wants to generate the inputs randomly
    {
        cout<<"Enter the number of cities:  (default 10)\n";
        //if user give 0 or negative number, it will be set to 10 by default (10 cities) and the program will run
        cin>>total_cities;
        if(total_cities<=0)
            total_cities=10;
        cout<<"Enter the range of the coordinates: (default 100)\n"; //enter the range of the coordinates (default 100)
        int range; //range of the coordinates
        cin>>range; //store the range of the coordinates
        if(range<=0)
            range=100;
        for(int i = 0;i<total_cities;i++) //generate the random coordinates for the cities
        {
            points[i][0] = generateRandomNumber(range); //generate the random coordinates for the x-coordinate
            points[i][1] = generateRandomNumber(range); //generate the random coordinates for the y-coordinate
        }
    }
    else if(choice == 3) //if the user wants to exit the program
        exit(0); //exit the program
    else
    {
        cout<<"Invalid choice! Try Again\n"; //if the user enters an invalid choice
        goto s; //go back to the start of the function to ask the user to give the inputs again.
    }
    for(int i = 0;i<total_cities;i++)
        for(int j = 0;j<total_cities;j++)
            originalGraph[i][j] = calculateDistance(points[i][0],points[i][1],points[j][0],points[j][1]); //calculate the distance between the cities

    if(total_cities < 15) { //if the number of cities is less than 15 because console window is too small to display the matrix correctly.
        cout<<"\n\t\tCOST MATRIX:\n";
        for (int i = 0; i < total_cities; ++i) //print the cost matrix
            cout<<"\tCity"<<i+1;
        for(int i = 0;i<total_cities;i++)
        {
            cout<<"\nCity"<<i+1<<" "; //print the city number
            for(int j = 0;j<total_cities;j++)
                cout << "\t" << originalGraph[i][j]; //print the distance between the cities
        }
        cout<<"\n\n";
    }
}

//insert the child in the population in the correct position based on the cost of the child and the cost
// of the other children in the population in ascending order of the cost of the children.
void insertBinarySearch(vector<int>& child, int total_cost)
{
    int i = 0;
    while (i < population.size() && population[i].second < total_cost) //find the position where the child should be inserted in the population
        i++;
    population.insert(population.begin() + i, make_pair(child, total_cost)); //insert the child in the population in the correct position
}

//check if the chromosome exists in the population or not and return true or false accordingly and also return the index of the chromosome
//function use std::any_of to check if the chromosome exists in the population or not and return true or false accordingly.
bool existsChromosome(const vector<int> & v)
{
    return std::any_of(population.begin(), population.end(), [&v](const pair<vector<int>, int> & p) { return p.first == v; });
}

int costCalculater(vector<int>& solution) //check if the solution is valid and if it is valid calculate cost of the solution and return the cost of the solution in the function call
{
    int total_cost=0; //initialize the total cost
    //store the solution
    vector<int> solution_copy = solution;
    //loop through the solution and check if the cities are connected or not and if they are connected then calculate the cost of the solution and return the cost
    for(int i = 0;i<total_cities;i++)
    {
        int j = solution_copy[i];
        int k = solution_copy[(i+1)%total_cities];
        if(originalGraph[j][k] == 0) //if the cities are not connected
        {
            return -1; //return -1 if the solution is not valid
        }
        else
        {
            total_cost += originalGraph[j][k]; //calculate the cost of the solution
        }
    }
    return total_cost; //return the cost of the solution
}

//sort the population in ascending order of the cost and return true or false accordingly
bool sortbysec(const pair<vector<int>, int> & one, const pair<vector<int>, int> & two)
{
    return one.second < two.second;
}

//create the initial population of the size of the population size and also check
// if the solution is valid or not and if it is valid then add it to the population and if it is not valid then discard it
void initialPopulation()
{
    vector<int> parent; //create a vector to store the parent
    parent.push_back(0); //insert the first city in the parent
    for(int i=1; i < total_cities; i++) //loop through the cities
        parent.push_back(i); //insert the cities in the parent
    int total_cost = costCalculater(parent); //check if the solution is valid or not
    cout<<"Initial cost= "<< total_cost<<endl;
    if(total_cost != -1) // checks if the parent is valid
    {
        population.emplace_back(parent, total_cost); //insert the parent in the population
        real_size_population++; //increment the real size of the population
    }
    for(int i=0;i<generations;i++) //loop through the generations
    {
        shuffle(parent.begin() + 1, parent.begin() + (generateRandomNumber(total_cities - 1) + 1), std::mt19937(std::random_device()())); //shuffle the parent
        total_cost = costCalculater(parent); //check if the solution is valid or not
        if(total_cost != -1) //checks if the parent is valid
        {
            population.emplace_back(parent, total_cost); //insert the parent in the population
            real_size_population++; //increment the real size of the population
        }
        if(real_size_population == total_cities) // checks size population
            break;

    }
    if(real_size_population == 0) //if the real size of the population is 0
        //cout << "\nEmpty initial population ;( Try again runs the algorithm...";

        sort(population.begin(), population.end(), sortbysec); //sort the population in ascending order of the cost
}

void crossOver(vector<int>& parent1, vector<int>& parent2) //cross over the parent1 and parent2
{
    vector<int> child1, child2; //create two children
    map<int, int> genes1, genes2; //create two maps to store the genes
    for(int i = 0; i < total_cities; i++) //loop through the cities
    {
        genes1[parent1[i]] = 0;  //insert the genes in the map
        genes2[parent2[i]] = 0; //insert the genes in the map
    }
    int point1 = generateRandomNumber(total_cities - 1) + 1; //get the point1 between 1 and total_cities-1
    int point2 = generateRandomNumber(total_cities - point1) + point1; //get the point2 between point1 and total_cities-1
    if(point1 == point2) //if the point1 and point2 are same then get the point2 between point1 and total_cities-1
    {
        if(point1 - 1 > 1) //if the point1 is greater than 1
            point1--; //decrement the point1
        else if(point2 + 1 < total_cities) //if the point2 is less than total_cities then increment the point2
            point2++; //increment the point2
        else
        {
            int point = generateRandomNumber(10) + 1; //get the point between 1 and 10
            if(point <= 5) //if the point is less than or equal to 5
                point1--; //decrement the point1 and point2
            else
                point2++;
        }
    }
    for(int i = 0; i < point1; i++) //loop through the point1 and insert the genes in the child1
    {
        // adds genes
        child1.push_back(parent1[i]); //insert the genes in the child1
        child2.push_back(parent2[i]); //insert the genes in the child2
        // marks genes
        genes1[parent1[i]] = 1; //insert the genes in the map
        genes2[parent2[i]] = 1; //insert the genes in the map
    }
    for(int i = point2 + 1; i < total_cities; i++) //loop through the point2 and insert the genes in the child1
    {
        genes1[parent1[i]] = 1; //insert the genes in the map
        genes2[parent2[i]] = 1; //insert the genes in the map
    }
    for(int i = point2; i >= point1; i--) //loop through the point2 and insert the genes in the child2
    {
        if(genes1[parent2[i]] == 0) // if the gene is not used
        {
            child1.push_back(parent2[i]); //insert the genes in the child1
            genes1[parent2[i]] = 1; // marks the gene
        }
        else
        {
            // if the gene already is used, chooses gene that is not used
            for(auto it = genes1.begin(); it != genes1.end(); ++it) //loop through the genes1
            {
                if(it->second == 0) // checks if is not used
                {
                    child1.push_back(it->first);
                    genes1[it->first] = 1; // marks as used
                    break; // left the loop
                }
            }
        }

        if(genes2[parent1[i]] == 0) // if the gene is not used
        {
            child2.push_back(parent1[i]); //insert the genes in the child2
            genes2[parent1[i]] = 1; // marks the gene
        }
        else
        {
            // if the gene already is used, chooses gene that is not used
            for(auto it = genes2.begin(); it != genes2.end(); ++it)
            {
                if(it->second == 0) // checks if is not used
                {
                    child2.push_back(it->first);
                    genes2[it->first] = 1; // marks as used
                    break; // left the loop
                }
            }
        }

    }
    for(int i = point2 + 1; i < total_cities; i++) //loop through the point2 and insert the genes in the child2
    { //insert the genes in the child2 and marks the genes in the map
        child1.push_back(parent1[i]);
        child2.push_back(parent2[i]);
    }
    int mutation = generateRandomNumber(100) + 1; //get the point between 1 and 100
    if(mutation <= mutation_rate) //if the point is less than or equal to mutation_rate
    {
        // makes a mutation: change of two genes

        int index_gene1, index_gene2;
        index_gene1 = generateRandomNumber(total_cities - 1) + 1; //get the point between 1 and total_cities - 1
        index_gene2 = generateRandomNumber(total_cities - 1) + 1; //get the point between 1 and total_cities - 1

        // makes for child1
        int aux = child1[index_gene1]; //save the gene in the auxiliar
        child1[index_gene1] = child1[index_gene2]; //change the gene in the index_gene1
        child1[index_gene2] = aux; //change the gene in the index_gene2

        // makes for child2
        aux = child2[index_gene1]; //save the gene in the auxiliar and change the gene in the index_gene1
        child2[index_gene1] = child2[index_gene2]; //change the gene in the index_gene2
        child2[index_gene2] = aux; //change the gene in the index_gene1
    }
    int total_cost_child1 = costCalculater(child1); //checks if the child1 is valid
//	printf("Initial Cost child 1=%d\total_cities",total_cost_child1);
    int total_cost_child2 = costCalculater(child2); //checks if the child2 is valid
    if(total_cost_child1 != -1 && !existsChromosome(child1)) //if the child1 is valid and does not exist
    {
        // add child in the population
        insertBinarySearch(child1, total_cost_child1); // uses binary search to insert
        real_size_population++; // increments the real_size_population
    }

    // checks again...
    if(total_cost_child2 != -1 && !existsChromosome(child2)) //if the child2 is valid and does not exist in the population
    {
        // add child in the population
        insertBinarySearch(child2, total_cost_child2); // uses binary search to insert in the population
        real_size_population++;
    }
}

void printPopulation() { //prints the population in the console
    for(int i = 0; i < real_size_population; i++) //loop through the population
    {
        cout << "Chromosome " << i+1 << ": "; //print the number of the chromosome
        const vector<int>& vec = population[i].first; //get the chromosome and save it in a vector
        for(int j = 0; j < total_cities; j++) //loop through the cities
            cout << vec[j]+1 << "-->"; //print the solution
        cout << vec[0]+1<<endl; //print the solution
        cout << " Cost: " << population[i].second<<endl; //print the cost of the chromosome
    }
}

void geneticRun() //run the genetic algorithm and print the results in the file "results.txt" and in the screen the results
{
    initialPopulation(); //initializes the population
    if(real_size_population==0) //if the population is empty
        return;
    for(int i=0;i<generations;i++) //loop through the generations
    {
        int old_size_population=real_size_population; //save the real_size_population
        if(real_size_population>=2) //if the population is bigger than 2 (the population is not empty)
        {
            if(real_size_population==2) //if the population is 2
            {
                crossOver(population[0].first, population[1].first); //crossOver the first and the second chromosome
            }
            else
            {
                int parent1,parent2;
                do{
                    parent1=generateRandomNumber(real_size_population); //get the point between 0 and real_size_population
                    parent2=generateRandomNumber(real_size_population); //get the point between 0 and real_size_population
                }while(parent1==parent2);
                crossOver(population[parent1].first, population[parent2].first); //crossOver the first and the second chromosome

            }
            int diff_population = real_size_population - old_size_population; //get the difference between the old_size_population and the real_size_population

            if(diff_population == 2) //if the difference is 2 (the population is full)
            {
                if(real_size_population > total_cities) //if the population is bigger than the total_cities
                {
                    // removes the two worst parents of the population
                    population.pop_back();
                    population.pop_back();

                    // decrements the real_size_population in 2 units
                    real_size_population -= 2;
                }
            }
            else if(diff_population == 1)
            {
                if(real_size_population > total_cities) //if the population is bigger than the total_cities (the population is not full)
                {
                    population.pop_back(); //removes the last element of the population (the worst chromosome)
                    real_size_population--; //decrements the real_size_population in 1 unit (the worst chromosome)
                }
            }
        }
        else //if the population is empty or 1 (the population is not full)
            initialPopulation(); //initializes the population
        //print the results in the file "results.txt"
        if (save_population_in_file) { //if the user wants to save the population in a file
            ofstream file;
            file.open("results.txt", ios::app);
            file << "Generation " << i << ": " << endl;
            for(int j=0;j<real_size_population;j++)
            {
                file << "Chromosome " << j << ": ";
                for(int k=0;k<total_cities;k++)
                {
                    file << population[j].first[k] << " ";
                }
                file << endl;
            }
            file << endl;
            file.close();
        }
        //print the results in the terminal
        if (show_population_in_terminal) { //if the user wants to show the population in the terminal
            if(i==0)
            {
                cout<<"The first generation is: "<<endl;
                printPopulation();
            }
            else if(i==generations-1)
            {
                cout<<"The last generation is: "<<endl;
                printPopulation();
            }
            else
            {
                cout<<"The "<<i+1<<" generation is: "<<endl;
                printPopulation();
            }
        }
    }
    //print the results in the file "results.txt"
    if (save_population_in_file) { //if the user wants to save the population in a file
        ofstream file;
        file.open("results.txt", ios::app);
        file << "The best chromosome is: " << endl;
        for(int j=0;j<total_cities;j++) {
            file << population[0].first[j] << " ";
        }
        file << endl;
        file.close();
    }
    cout << "\nBest solution: ";
    const vector<int>& vec = population[0].first; //get the first chromosome of the population (the best chromosome) and save it in a vector
    for(int i = 0; i < total_cities; i++) //loop through the cities
        cout << vec[i]+1 << "-->"; //print the solution
    cout << vec[0]+1<<endl; //print the solution
    cout << " Minimum Cost: " << population[0].second<<endl; //print the cost of the best chromosome
}


int main()
{
    get();

    for(int i = 0; i <3; i++) // runs the genetic algorithm 3 times
    {
        cout << "\n\n\nRUN " << i+1 << endl;
        clock_t begin_time=clock();
        geneticRun();
        cout << "Time spent: " << (double)(clock()-begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
        //print the results in the file "results.txt"
        if (save_population_in_file) { //if the user wants to save the population in a file
            ofstream file;
            file.open("results.txt", ios::app);
            file << "Run " << i+1 << ": " << endl;
            file << "Time spent: " << (double)(clock()-begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
            file.close();
        }
        //print the real_size_population
        cout << "Real size of the population: " << real_size_population << endl;
    }
    return 0;
}
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "17021997",
   "metadata": {},
   "source": [
    "# Expectation Decider – Probability & Statistics\n",
    "This notebook solves the assignment step by step using a generated dataset of 200 students."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6d45cdae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>study_hours</th>\n",
       "      <th>attendance</th>\n",
       "      <th>group_discussion</th>\n",
       "      <th>previous_test_score</th>\n",
       "      <th>final_exam_pass</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14.0</td>\n",
       "      <td>82.3</td>\n",
       "      <td>Yes</td>\n",
       "      <td>70</td>\n",
       "      <td>Pass</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11.4</td>\n",
       "      <td>84.7</td>\n",
       "      <td>Yes</td>\n",
       "      <td>59</td>\n",
       "      <td>Pass</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>14.6</td>\n",
       "      <td>91.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>58</td>\n",
       "      <td>Pass</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>18.1</td>\n",
       "      <td>90.6</td>\n",
       "      <td>No</td>\n",
       "      <td>59</td>\n",
       "      <td>Fail</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>11.1</td>\n",
       "      <td>61.5</td>\n",
       "      <td>No</td>\n",
       "      <td>71</td>\n",
       "      <td>Fail</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   study_hours  attendance group_discussion  previous_test_score  \\\n",
       "0         14.0        82.3              Yes                   70   \n",
       "1         11.4        84.7              Yes                   59   \n",
       "2         14.6        91.0              Yes                   58   \n",
       "3         18.1        90.6               No                   59   \n",
       "4         11.1        61.5               No                   71   \n",
       "\n",
       "  final_exam_pass  \n",
       "0            Pass  \n",
       "1            Pass  \n",
       "2            Pass  \n",
       "3            Fail  \n",
       "4            Fail  "
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from math import comb\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "df = pd.read_csv('expectation_decider_dataset.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "819b78cc",
   "metadata": {},
   "source": [
    "## 1. Understanding the Basics\n",
    "**Probability** is the numerical chance that an event will happen.\n",
    "\n",
    "Examples from this dataset:\n",
    "1. Probability that a randomly selected student passes.\n",
    "2. Probability that a student studies more than 10 hours per week.\n",
    "3. Probability that a student participates in group discussion."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1e29bd21",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "P(Pass) = 0.495\n",
      "P(Study hours > 10) = 0.685\n",
      "P(Group discussion = Yes) = 0.605\n"
     ]
    }
   ],
   "source": [
    "n = len(df)\n",
    "p_pass = (df['final_exam_pass'] == 'Pass').mean()\n",
    "p_study_above_10 = (df['study_hours'] > 10).mean()\n",
    "p_group_yes = (df['group_discussion'] == 'Yes').mean()\n",
    "print('P(Pass) =', p_pass)\n",
    "print('P(Study hours > 10) =', p_study_above_10)\n",
    "print('P(Group discussion = Yes) =', p_group_yes)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6255acbd",
   "metadata": {},
   "source": [
    "## 2. Empirical and Theoretical Probability\n",
    "**Empirical probability** is calculated from observed data.\n",
    "\n",
    "**Theoretical probability** is calculated using a mathematical model. For 3 independent selections where the probability of passing is `p`, the number of passes follows a binomial distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d704792e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empirical probability of passing = 0.495\n",
      "Theoretical/model probability of exactly 2 passes out of 3 = 0.37121287499999994\n"
     ]
    }
   ],
   "source": [
    "empirical_probability = (df['final_exam_pass'] == 'Pass').mean()\n",
    "print('Empirical probability of passing =', empirical_probability)\n",
    "\n",
    "p = empirical_probability\n",
    "k = 2\n",
    "n_trials = 3\n",
    "theoretical_probability_exactly_2 = comb(n_trials, k) * (p ** k) * ((1-p) ** (n_trials-k))\n",
    "print('Theoretical/model probability of exactly 2 passes out of 3 =', theoretical_probability_exactly_2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4381c278",
   "metadata": {},
   "source": [
    "## 3. Random Variable and Probability Distribution\n",
    "Let **X = number of students passing among 3 randomly selected students**.\n",
    "\n",
    "X can take values 0, 1, 2, or 3. We use the binomial model with `p = P(Pass)`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8bb1757f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   X      P(X)\n",
      "0  0  0.128788\n",
      "1  1  0.378712\n",
      "2  2  0.371213\n",
      "3  3  0.121287\n",
      "Mean = 1.4849999999999999\n",
      "Variance = 0.749925\n"
     ]
    }
   ],
   "source": [
    "x_values = np.arange(0, 4)\n",
    "probabilities = [comb(3, x) * p**x * (1-p)**(3-x) for x in x_values]\n",
    "dist = pd.DataFrame({'X': x_values, 'P(X)': probabilities})\n",
    "print(dist)\n",
    "mean_x = 3 * p\n",
    "variance_x = 3 * p * (1-p)\n",
    "print('Mean =', mean_x)\n",
    "print('Variance =', variance_x)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "447d2800",
   "metadata": {},
   "source": [
    "## 4. Venn Diagram Conditions\n",
    "A = students who study more than 10 hours/week.\n",
    "\n",
    "B = students whose attendance is greater than 80%.\n",
    "\n",
    "The overlap A ∩ B contains students satisfying both conditions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9db9782e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A only          69\n",
      "B only          25\n",
      "Both (A ∩ B)    68\n",
      "Neither         38\n",
      "dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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9rZdu3aZ6OjoDDpC90JwymS+/PJLkzt3brNjxw5jjDFr1641vr6+ZsiQIfY+LVu2NB06dHDY77fffjOSzIEDB4wxxpw/f954e3ubFStWJHmNrFmzmp9//jndjgGpM3fuXJM/f36za9cuY4wx33//vcmePbt5++237X2aNm1qunTp4rDf5s2bHX7pnT592nh6epq1a9c69EtISDA2m81s3LgxfQ8ELtW4cWMzYMAAh7amTZuaPn36pLjP9evXTUBAgBk1apT55ZdfTPfu3U2uXLnM2LFj07vcTCc2Ntb+7/MHH3xgihUrZvLmzWuuXLli77N161YjyR6w7ggLCzNdu3Y1xty+XPf0008bPz8/88orr5iBAweaEiVKmN69eyf5T1NmRXB6wF29etV8/fXX5saNG5b616xZ0wwePNihbeLEiSZbtmzmwoULxhhjli5darJly2ZiY2Md+pUvX96MHDnS/vynn34yly9ftj+Pj483//nPfxzGQtq7evWq+eabb8yIESPMxIkTzaVLl+7a//HHHzdvvfWWQ9vYsWNNzpw57edv0aJFJkeOHA7/yBpjTHBwsMP/Mn/88UeHPtevXzfvv/++w1hwf7///rvp2rWrWb9+veV9atSoYSZPnuzQtm7dumT/3PxTnz59jCTz6KOPmvfff9+cOXPG6bqRsoULFxpfX19z9epVY4wxf/31l5FkNm/e7NCvbNmy5p133nFo++KLL0zOnDnta5lu3bpl5syZY15++WUzYsQIs3Xr1ow5iAcEwekB17p1ayPJdOvWzVL/cuXKOcw0GGPMpUuXjIeHh/3yXHx8vAkMDDSff/65Q7/Zs2ebWbNmpTj26tWrTb169UxERERqDgEWJCYmmp9++sl0797dBAYGmgYNGphXXnnFFCxY0FSvXt0kJiamuG/x4sXNuHHjHNqio6ONJPuM4bVr14y/v7+ZM2eOQ7/PP/88Sds/LV++3NSvX5/ZJjd1+vRps2jRoiTtb7zxhvH19TWBgYFm3759lsbq3r276d69u0NbYmKiKV68+F3/jPz1118Ol31w/+Li4sxff/3l8B/mK1eumJw5c5oFCxbY22rVqmX69+/vsO+YMWNMyZIlHdquXr1qfH19+VCPRQSnB1zTpk1Nt27dTN68eZPMKiSnTZs2pl69eknaCxYs6PC/yb59+ybbD65x5x+2Zs2a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",
      "text/plain": [
       "<Figure size 600x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "A = df['study_hours'] > 10\n",
    "B = df['attendance'] > 80\n",
    "counts = {\n",
    "    'A only': (A & ~B).sum(),\n",
    "    'B only': (~A & B).sum(),\n",
    "    'Both (A ∩ B)': (A & B).sum(),\n",
    "    'Neither': (~A & ~B).sum()\n",
    "}\n",
    "print(pd.Series(counts))\n",
    "\n",
    "plt.figure(figsize=(6,4))\n",
    "plt.bar(counts.keys(), counts.values())\n",
    "plt.xticks(rotation=20)\n",
    "plt.ylabel('Number of students')\n",
    "plt.title('Venn-Condition Counts')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f821587e",
   "metadata": {},
   "source": [
    "## 5. Contingency Table and Probability Calculations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3fd91d2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final_exam_pass   Fail  Pass\n",
      "group_discussion            \n",
      "No                  47    32\n",
      "Yes                 54    67\n",
      "Joint P(Group discussion Yes AND Pass) = 0.335\n",
      "Marginal P(Pass) = 0.495\n",
      "Conditional P(Pass | Group discussion Yes) = 0.5537190082644629\n"
     ]
    }
   ],
   "source": [
    "table = pd.crosstab(df['group_discussion'], df['final_exam_pass'])\n",
    "print(table)\n",
    "\n",
    "total = len(df)\n",
    "joint_probability = ((df['group_discussion'] == 'Yes') & (df['final_exam_pass'] == 'Pass')).mean()\n",
    "marginal_probability_pass = (df['final_exam_pass'] == 'Pass').mean()\n",
    "conditional_probability = (\n",
    "    ((df['group_discussion'] == 'Yes') & (df['final_exam_pass'] == 'Pass')).sum()\n",
    "    / (df['group_discussion'] == 'Yes').sum()\n",
    ")\n",
    "print('Joint P(Group discussion Yes AND Pass) =', joint_probability)\n",
    "print('Marginal P(Pass) =', marginal_probability_pass)\n",
    "print('Conditional P(Pass | Group discussion Yes) =', conditional_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a27c103",
   "metadata": {},
   "source": [
    "## 6. Conditional Probability and Independence\n",
    "If P(Pass | Group discussion = Yes) is different from P(Pass), the two events are not independent.\n",
    "\n",
    "They are not mutually exclusive because a student can participate in group discussion and pass the exam at the same time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "caea5975",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "P(Pass | Yes) = 0.5537190082644629\n",
      "P(Pass) = 0.495\n",
      "Independent? False\n"
     ]
    }
   ],
   "source": [
    "p_pass_given_yes = conditional_probability\n",
    "print('P(Pass | Yes) =', p_pass_given_yes)\n",
    "print('P(Pass) =', marginal_probability_pass)\n",
    "print('Independent?', np.isclose(p_pass_given_yes, marginal_probability_pass))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef6c881e",
   "metadata": {},
   "source": [
    "## 7. Bayes Theorem\n",
    "Given:\n",
    "- P(High attendance | Pass) = 0.70\n",
    "- P(High attendance | Fail) = 0.40\n",
    "- P(High attendance) = 0.60\n",
    "\n",
    "Using Bayes theorem:\n",
    "P(Pass | High attendance) = P(High attendance | Pass) × P(Pass) / P(High attendance)\n",
    "\n",
    "The assignment does not explicitly provide P(Pass), so we estimate it from the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "735948bd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Estimated P(Pass | High attendance) = 0.5775\n"
     ]
    }
   ],
   "source": [
    "p_high_given_pass = 0.70\n",
    "p_high_given_fail = 0.40\n",
    "p_high = 0.60\n",
    "p_pass_dataset = (df['final_exam_pass'] == 'Pass').mean()\n",
    "bayes_result = (p_high_given_pass * p_pass_dataset) / p_high\n",
    "print('Estimated P(Pass | High attendance) =', bayes_result)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3ebe772",
   "metadata": {},
   "source": [
    "## Final Summary\n",
    "The analysis compares study hours, attendance, group discussion, and previous test scores with exam outcomes. The generated dataset is illustrative; results depend on the random seed and should not be interpreted as real educational evidence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec7c7482-6f98-4166-84f0-3978101b57cf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4b489fc-a8cc-4424-a008-8c4616602596",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

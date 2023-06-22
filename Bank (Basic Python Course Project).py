#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class menu:
    @staticmethod
    def run_bank():
        menu_return = 0
        while menu_return == 0:
            user_action = input("Select from the following options:\n\n 1) Open New Account \n 2) Check Account Info \n 3) Deposit Funds \n 4) Withdraw Funds \n 5) Quick Balance \n\n Select from 1-5 or enter Q to quit at any time: ")
            if user_action.lower() == 'q':
                print("Exiting bank program...")
                import time
                time.sleep(3)
                print("Done.")
                break
# Open a new Account.
            elif user_action == '1':
                menu_return += 1
                menu_reset = ""
                while menu_reset == "":
                    account_type = input("\n Checking (1), savings (2), or quit (Q)? ")
                    if account_type == '1':
                        temp_account = Account()                       
                        temp_account.enterAccountInfo()
                        menu_reset = 1
                    elif account_type == '2':
                        temp_account = SavingsAccount()
                        temp_account.enterSavingsAccountInfo()
                        menu_reset = 1
                    elif account_type.lower() == 'q':
                        menu_reset = 1
                    else:
                        menu_reset = ""
                menu_return = 0
# Get existing account info.
            elif user_action == '2':
                menu_return += 1
                menu_reset = ""
                while menu_reset == "":
                    account_id_input = input("\n Enter your Account ID: ")
                    print("\n Enter Q to quit at any time. \n")
                    match = 0
                    if account_id_input.lower() == 'q':
                        break
                    for acct_id in Account.accounts:
                        # print(acct_id.AccountID)
                        # print(acct_id.account_type)
                            if str(acct_id.AccountID) == account_id_input:
                                match = 1
                                if str(acct_id.account_type).lower() == 'checking':
                                    acct_id.getAccountInfo()
                                    menu_reset = 1
                                elif str(acct_id.account_type).lower() == 'savings':
                                    acct_id.getSavingsAccountInfo()
                                    menu_reset = 1
                                elif account_id_input.lower() == 'q':
                                    menu_reset = 1
                                else:
                                    pass
                    if match == 0:
                        print("Invalid Account. Please try again. \n")
                        menu_reset = ""  
                menu_return = 0
            elif user_action in ['3', '4', '5']:
                account_id_input = ""
                while account_id_input == "":
                    account_id_input = input("\n Enter your Account ID: ")
                    if account_id_input.lower() == 'q':
                        break
                    for acct_id in Account.accounts:
                        match = 0
                        if str(acct_id.AccountID) == account_id_input:
                            if user_action == '3':
                                match += 1
                                if str(acct_id.account_type).lower() == 'checking':
                                    acct_id.deposit(account_id_input)
                                    acct_id.getAccountInfo()
                                    break
                                elif str(acct_id.account_type).lower() == 'savings':
                                    acct_id.depositSavingsAcct(account_id_input)
                                    acct_id.getSavingsAccountInfo()
                                    break
                                elif account_id_input.lower() == 'q':
                                    break
                                else:
                                    pass
                            elif user_action == '4':
                                match += 1
                                if str(acct_id.account_type).lower() == 'checking':
                                    acct_id.withdraw(account_id_input)
                                    break
                                elif str(acct_id.account_type).lower() == 'savings':
                                    acct_id.withdrawSavingsAcct(account_id_input)
                                    break
                                elif account_id_input.lower() == 'q':
                                    break
                                else:
                                    pass
                            elif user_action == '5':
                                match += 1
                                acct_id.getBalance(account_id_input)
                            break
                    if match == 0:
                        print("Invalid Account. Please try again. \n")
                        account_id_input = ""
            else:
                print("Invalid input.")
                menu_return = 0

            
# Initialize ID counters.
base_account_id = 100000
base_customer_id = 10000000

# Initialize Bank Class.
class Bank:
    def __init__(self, IFSC_Code, bankname, branchname, loc):
        self.IFSC_Code = IFSC_Code
        self.bankname = bankname
        self.branchname = branchname
        self.loc = loc

    def print_info(self):
        print('Bank Information :')
        print('IFSC_Code : ', self.IFSC_Code)
        print('Bank Name : ', self.bankname)
        print('Branch Name : ', self.branchname)
        print('Location : ', self.loc)

# Initialize Customer class inheriting from Bank.
class Customer:
    def __init__(self, CustomerID, custname, address, contactdetails):
        self.CustomerID = CustomerID
        self.custname = custname
        self.address = address
        self.contactdetails = contactdetails

    def print_customer_info(self):
        print('Customer Information :')
        print('Customer ID : ', self.CustomerID)
        print('Customer Name : ', self.custname)
        print('Address : ', self.address)
        print('Contact Details : ', self.contactdetails)

# Initialize Account class inheriting from Bank and Customer.
class Account(Bank, Customer):
    accounts = []
    def __init__(self, IFSC_Code=55555, bankname='Fifth Third', branchname='Chicago', loc='Chicago, IL', CustomerID='', custname='', address='', contactdetails='', AccountID='', balance=0, account_type='checking'):   
        Bank.__init__(self, IFSC_Code, bankname, branchname, loc)
        Customer.__init__(self, CustomerID, custname, address, contactdetails)
        self.AccountID = AccountID
        self.balance = balance
        self.account_type = account_type

   # Look up account information.
    def getAccountInfo(self):
        print('\nAccount Information :')
        print('Account ID : ', self.AccountID)
        print('Customer Name : ', self.custname)
        print('Balance : ', self.balance)

   # Add new checking account.
    def enterAccountInfo(self, bankname='Fifth Third', branchname='Chicago', loc='Chicago, IL', CustomerID='', custname='', address='', contactdetails='', balance=0, account_type='checking', IFSC_Code=55555):
        global base_account_id
        global base_customer_id
        base_customer_id += 1
        base_account_id += 1
        """IFSC_Code = input("Enter IFSC code: ")
        bankname = input("Enter bank name: ")
        branchname = input("Enter branch name: ")
        loc = input("Enter location: ")"""
        AccountID = base_account_id
        CustomerID = base_customer_id
        custname = input("\nEnter customer name: ")
        address = input("Enter customer address: ")
        contactdetails = input("Enter customer contact details: ")
        balance = ""
        account_type = "Checking"
        while balance == "":
            try:
                balance = float(input("Initial deposit: "))
                if balance < 0:
                    raise ValueError("Please enter a positive dollar value.")
            except ValueError:
                print("Invalid input. Please enter a positive dollar value.")
                balance = ""
        self = Account(IFSC_Code, bankname, branchname, loc, CustomerID, custname, address, contactdetails, AccountID, balance, account_type)
        Account.accounts.append(self)
        self.getAccountInfo()
        return
    
    # Check if account exists.
    def checkAccountID(self, account_id_input):
        match_found = False
        for acct in Account.accounts:
            if str(acct.AccountID) == account_id_input:
                match_found = True
                break
        return match_found
    
    # Deposit money.
    def deposit(self,account_id_input):
        menu_exit = 0
        if account_id_input == "" :
            account_id_input = input("\n Enter your Account ID: ")
        else:
            pass
        self.checkAccountID(account_id_input)
        if not self.checkAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            account_id_input = ""
        else:
            amount = ""
        while amount == "":
            amount = input("\nEnter 0 to return to the menu.\nDeposit amount: ")
            try:
                amount = float(amount)
                if amount < 0:
                    raise ValueError("Please enter a positive dollar value.")
            except ValueError:
                print("Invalid input. Please enter a positive dollar value.")
                amount = ""
                continue
            for acct_id in Account.accounts:
                if str(acct_id.AccountID) == account_id_input:
                    acct_id.balance = float(acct_id.balance) + amount
                    print("\nAccount: " + str(account_id_input))
                    print("New Balance: " + str(acct_id.balance))
                    break
            else:
                print("Account not found. Please try again.")
                account_id_input = ""
                amount = ""
                continue
    
    #Intermediate: check account balance.
    def testBalance(self, account_id_input):
        for acct in Account.accounts:
            if str(acct.AccountID) == account_id_input:
                return acct.balance
    
    #Intermediate: check account balance.
    def getBalance(self,account_id_input):
        if account_id_input == "" :
            account_id_input = input("\n Enter your Account ID: ")
        else:
            pass
        if not self.checkAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            account_id_input = ""
        else:
            for acct_id in Account.accounts:
                if str(acct_id.AccountID) == account_id_input:
                    print("\nAccount: " + str(account_id_input))
                    print("Balance: " + str(acct_id.balance))

    # Withdraw money.
    def withdraw(self, account_id_input):
        if account_id_input == "" :
            account_id_input = input("\n Enter your Account ID: ")
        else:
            pass
        if not self.checkAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            account_id_input = ""
        else:
            amount = ""
            while amount == "":
                amount = input("\nEnter 0 to return to the menu.\nWithdrawal amount: ")
                try:
                    amount = float(amount)
                    if amount < 0:
                        raise ValueError("Please enter a positive dollar value.")
                except ValueError:
                    print("Invalid input. Please enter a positive dollar value.")
                    amount = ""
                    continue
                for acct_id in Account.accounts:
                    if str(acct_id.AccountID) == account_id_input:
                        if self.testBalance(account_id_input) >= float(amount):
                            acct_id.balance = acct_id.balance - float(amount)
                            print("\nAccount: " + str(account_id_input))
                            print("New Balance: " + str(acct_id.balance))
                        else:
                            print("Insufficient funds. Please reduce the withdrawal. The current balance on the account is "+str(acct_id.balance))
                            amount = ''

class SavingsAccount(Account):
    def __init__(self, custname='', address='', contactdetails='', balance=0, IFSC_Code=55555, bankname='Fifth Third', branchname='Chicago', loc='Chicago, IL', minbalance=500, account_type='Savings', CustomerID='', AccountID=''):
        super().__init__(custname=custname, address=address, contactdetails=contactdetails, balance=balance, account_type=account_type, CustomerID=CustomerID, AccountID=AccountID)
        self.minbalance = minbalance
        self.IFSC_Code = IFSC_Code
        self.bankname = bankname
        self.branchname = branchname
        self.loc = loc

    # Look up account information.
    def getSavingsAccountInfo(self):
        print('\nAccount Information :')
        print('Account ID : ', self.AccountID)
        print('Customer Name : ', self.custname)
        print('Balance : ', self.balance)
        print('Minimum Balance : ', self.minbalance)

   # Add new savings account.
    def enterSavingsAccountInfo(self, custname='', address='', contactdetails='', balance=0, IFSC_Code=55555, bankname='Fifth Third', branchname='Chicago', loc='Chicago, IL', minbalance=500):
        global base_account_id
        global base_customer_id
        base_account_id += 1
        base_customer_id += 1
        custname = input("\nEnter customer name: ")
        address = input("Enter customer address: ")
        contactdetails = input("Enter customer contact details: ")
        minbalance = 500
        account_type = "Savings"
        balance = None
        while balance is None:
            try:
                balance = float(input("Initial deposit: "))
                if balance < minbalance:
                    print(f"The balance must exceed the minimum balance of ${minbalance}.")
                    balance = None
            except ValueError:
                print("Invalid input. Please enter a positive dollar value.")
        if balance is not None:
            self = SavingsAccount(custname, address, contactdetails, balance, IFSC_Code, bankname, branchname, loc, minbalance, 'Savings', base_customer_id, base_account_id)
            SavingsAccount.accounts.append(self)
            self.getSavingsAccountInfo()
        return
    
    
    # Check if account exists.
    def checkSavingsAccountID(self, account_id_input):
        for acct in Account.accounts:
            if str(acct.AccountID) == account_id_input:
                return True
        print("Invalid Account ID. Please try again.")
        return False
    
    # Deposit to Savings Account.
    def depositSavingsAcct(self, account_id_input):
        menu_exit = 0
        if account_id_input == "" :
            account_id_input = input("\n Enter your Account ID: ")
        else:
            pass
        if not self.checkSavingsAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            account_id_input = ""
        else:
            """if not self.checkSavingsAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            return"""
            amount = ""
            while amount == "":
                amount = input("\nEnter 0 to return to the menu.\nAmount: ")
                try:
                    amount = float(amount)
                    if amount < 0:
                        raise ValueError("Please enter a positive dollar value.")
                except ValueError:
                    print("Invalid input. Please enter a positive dollar value.")
                    amount = ""

            for acct_id in Account.accounts:
                if str(acct_id.AccountID) == account_id_input:
                    acct_id.balance += amount
                    print("\nAccount: " + str(account_id_input))
                    print("New Balance: " + str(acct_id.balance))
                    break
            else:
                print("Account not found. Please try again.")
    
    # Withdraw money.
    def withdrawSavingsAcct(self,account_id_input):
        if account_id_input == "" :
            account_id_input = input("\n Enter your Account ID: ")
        else:
            pass
        if not self.checkAccountID(account_id_input):
            print("Invalid Account ID. Please try again.")
            account_id_input = ""
        else:
            amount = ""
            while amount == "":
                amount = input("\nEnter 0 to return to the menu. \nWithdrawal amount: ")
                try:
                    amount = float(amount)
                    if amount < 0:
                        raise ValueError("Please enter a positive dollar value.")
                except ValueError:
                    print("Invalid input. Please enter a positive dollar value.")
                    amount = ""
                for acct_id in Account.accounts:
                    if str(acct_id.AccountID) == account_id_input:
                        if self.testBalance(account_id_input) >= amount and (acct_id.balance - amount) >= acct_id.minbalance:
                            acct_id.balance = acct_id.balance - amount
                            print("\nAccount: " + str(account_id_input))
                            print("New Balance: " + str(acct_id.balance))
                        else:
                            print("Insufficient funds. Please reduce the withdrawal. Please note, the remaining balance in the account must be above " + str(acct_id.minbalance) + ". The current balance on the account is " + str(acct_id.balance))
                            amount = ''
                            continue
                            
# Initialize bank, customer, and account objects
test_bank = Bank('sample_code', 'Fifth Third Bank', 'Test Branch', 'Los Angeles, CA')
test_customer1 = Customer('12345', 'Bill Jones', '123 123rd St. Boston, MA', 'Cell Phone: 555-555-5555')
test_customer2 = Customer('67890', 'Jane Doe', '456 456th St. New York, NY', 'Cell Phone: 555-555-5556')
# test_account1 = Account('sample_code', 'Fifth Third Bank', 'Dunder Mifflin', 'Scranton, PA', '12345', 'Bill Jones', '123 123rd St. Boston, MA', 'Cell Phone: 555-555-5555',75839, 225.58, "checking")
# test_account2 (DO NOT USE) = SavingsAccount('67890', 'Jane Doe', '456 456th St. New York, NY', 'Cell Phone: 555-555-5556', 85903, 1000.00, 500, "savings")
# test_account3 = SavingsAccount('sample_code', 'Fifth Third Bank', 'Dunder Mifflin', 'Scranton, PA', '12345', 'Bill Jones', '123 123rd St. Boston, MA', 'Cell Phone: 555-555-5555',75903, 500.00, 500, "savings")

# Add test accounts to accounts object.
# Account.accounts.append(test_account1)
# SavingsAccount.accounts.append(test_account2)


bank_menu = menu()
bank_menu.run_bank()


# In[ ]:




